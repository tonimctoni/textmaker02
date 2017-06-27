#if defined(_OPENMP)
    #include "omp_matrix.hpp"
#endif
#include "tanh_layer.hpp"
#include "lstm_layer.hpp"
#include "softmax_layer.hpp"
#include <unordered_map>
#include <memory>
#include <tuple>
#include <chrono>
#include <random>
using namespace std;

template<typename T>
void get_type(T t);

class NeuralNetwork
{
private:
    static constexpr unsigned long time_steps=250;
    static constexpr unsigned long batch_size=23;
    static constexpr double learning_rate_layer1=0.0001;
    static constexpr double learning_rate_layer2=0.001;
    static constexpr double learning_rate_layer3=0.0001;
    static constexpr double decay=0.9;
    #define asoiaf_filename "../asoiaf/asoiaf.txt"
    #define allowed_chars "! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz"
    static constexpr unsigned long allowed_char_amount=46;
    static constexpr unsigned long lstm_output_size=200;

    const string index_to_char;
    unordered_map<char, size_t> char_to_index;
    string asoiaf_content;
    std::mt19937 gen;
    std::uniform_int_distribution<size_t> dst_start;

    array<OneHots<batch_size,allowed_char_amount>, time_steps> Xs;
    TanhLayerRMSProp<allowed_char_amount, allowed_char_amount/4, batch_size, time_steps> layer1;
    LstmLayerRMSProp<allowed_char_amount/4, lstm_output_size, batch_size, time_steps> layer2;
    SoftmaxLayerRMSProp<lstm_output_size, allowed_char_amount, batch_size, time_steps> layer3;
    array<OneHots<batch_size,allowed_char_amount>, time_steps> Ys;

    void set_XY()
    {
        for(size_t batch=0;batch<batch_size;batch++)
        {
            size_t start=dst_start(gen);
            for(size_t time_step=0;time_step<time_steps;time_step++)
            {
                Xs[time_step].set(batch, char_to_index[asoiaf_content[start+time_step]]);
                Ys[time_step].set(batch, char_to_index[asoiaf_content[start+time_step+1]]);
            }
        }
    }

    void calc()
    {
        for(size_t time_step=0;time_step<time_steps;time_step++)
        {
            layer1.calc(Xs[time_step].get(), time_step);
            layer2.calc(layer1.get_output(time_step), time_step);
            layer3.calc(layer2.get_output(time_step), time_step);
        }
    }

    void set_deltas()
    {
        for(size_t time_step=time_steps-1;;time_step--)
        {
            layer3.set_first_delta_and_propagate_with_cross_enthropy(Ys[time_step].get(), layer2.get_delta_output(time_step), time_step);
            layer2.propagate_delta(layer1.get_delta_output(time_step), time_step);
            layer1.propagate_delta(time_step);
            if(time_step==0) break;
        }
    }

    void learn()
    {
        for(size_t time_step=0;time_step<time_steps;time_step++)
        {
            layer1.update_weights_with_rmsprop(Xs[time_step].get(), time_step, learning_rate_layer1, decay);
            layer2.update_weights_with_rmsprop(layer1.get_output(time_step), time_step, learning_rate_layer2, decay);
            layer3.update_weights_with_rmsprop(layer2.get_output(time_step), time_step, learning_rate_layer3, decay);
        }
    }
public:
    NeuralNetwork():
    index_to_char(allowed_chars),asoiaf_content(get_file_content_as_string(asoiaf_filename))
    , gen(random_device()()),dst_start(0,asoiaf_content.size()-time_steps-1-1)
    ,layer1(),layer2(),layer3()
    {
        for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
        assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);
    }

    void print_info() const noexcept
    {
        print("Info:");
        print("time_steps:", time_steps);
        print("batch_size:", batch_size);
        print("learning_rates:", learning_rate_layer1, learning_rate_layer2, learning_rate_layer3);
        print("lstm_output_sizes:", lstm_output_size);
        print();
    }

    auto get_layers() const -> decltype(tie(layer1, layer2, layer3))
    {
        return tie(layer1, layer2, layer3);
    }

    void show_guts() const noexcept
    {
        layer1.show_guts();
        layer2.show_guts();
        layer3.show_guts();
    }

    void pre_train_layer1()
    {
        uniform_int_distribution<size_t> dst(0,allowed_char_amount-1);
        unique_ptr<SoftmaxLayerRMSProp<allowed_char_amount/4, allowed_char_amount, batch_size, 1>>
        layer2(new SoftmaxLayerRMSProp<allowed_char_amount/4, allowed_char_amount, batch_size, 1>);

        unique_ptr<OneHots<batch_size,allowed_char_amount>> X(new OneHots<batch_size,allowed_char_amount>);
        double error=1000.0;
        for(size_t iteration=0;error>1.0e-07;iteration++)
        {
            for(size_t i=0;i<batch_size;i++) X->set(i,dst(gen));

            layer1.calc(X->get(),0);
            layer2->calc(layer1.get_output(0),0);

            layer2->set_first_delta_and_propagate_with_cross_enthropy(X->get(), layer1.get_delta_output(0), 0);
            layer1.propagate_delta(0);

            layer1.update_weights_with_rmsprop(X->get(), 0, 0.02, decay);
            layer2->update_weights_with_rmsprop(layer1.get_output(0), 0, 0.02, decay);

            if(iteration%10000==0)
            {
                error=0.0;
                for(size_t i=0;i<batch_size;i++)
                {
                    for(size_t j=0;j<allowed_char_amount;j++)
                    {
                        error+=(layer2->get_output(0)[i][j]-X->get()[i][j])*(layer2->get_output(0)[i][j]-X->get()[i][j]);
                    }
                }
                error=sqrt(error);
                print(error);
            }
        }
    }

    double get_error() noexcept
    {
        double error=0.0;
        for(size_t time_step=0;time_step<time_steps;time_step++)
        {
            error+=layer3.get_delta_output(time_step).sum_of_squares();
        }
        return error/(batch_size*time_steps*allowed_char_amount);
    }

    void iterate()
    {
        set_XY();
        calc();
        set_deltas();
        learn();
    }
};

template<unsigned long A, unsigned long B, unsigned long C, unsigned long D, unsigned long E>
void produce_output(tuple<TanhLayerRMSProp<A, B, C, D> const&, LstmLayerRMSProp<B, E, C, D> const&, SoftmaxLayerRMSProp<E, A, C, D> const&> t)
{
    static constexpr unsigned long time_steps=512;
    static constexpr unsigned long allowed_char_amount=A;
    unique_ptr<TanhLayerBase<A, B, 1, time_steps>> layer1(new TanhLayerBase<A, B, 1, time_steps>);
    unique_ptr<LstmLayerBase<B, E, 1, time_steps>> layer2(new LstmLayerBase<B, E, 1, time_steps>);
    unique_ptr<SoftmaxLayerBase<E, A, 1, time_steps>> layer3(new SoftmaxLayerBase<E, A, 1, time_steps>);
    // TanhLayerBase<A, B, 1, time_steps> layer1;
    // LstmLayerBase<B, E, 1, time_steps> layer2;
    // SoftmaxLayerBase<E, A, 1, time_steps> layer3;
    layer1->set_wb(get<0>(t));
    layer2->set_wb(get<1>(t));
    layer3->set_wb(get<2>(t));

    const string index_to_char="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz";
    OneHots<1,allowed_char_amount> X;
    X.set(0, index_to_char.find(' '));
    for(size_t time_step=0;time_step<time_steps;time_step++)
    {
        layer1->calc(X.get(), time_step);
        layer2->calc(layer1->get_output(time_step), time_step);
        layer3->calc(layer2->get_output(time_step), time_step);
        size_t new_char_index=get_weighted_random_index(layer3->get_output(time_step)[0]);
        X.set(0, new_char_index);
        cout << index_to_char[new_char_index];
    }
    cout << "\n\n" << endl;
}

double elapsed_seconds() noexcept
{
    using namespace std::chrono;
    static auto last = steady_clock::now();
    auto now=steady_clock::now();
    auto elapsed = duration_cast<milliseconds>(now-last).count();
    last=now;
    return elapsed/1000.;
}

int main()
{
    static constexpr unsigned long iterations=-1;
    static constexpr unsigned long output_period=20;
    unique_ptr<NeuralNetwork> neural_network(new NeuralNetwork);
    neural_network->pre_train_layer1();
    elapsed_seconds();
    double error=0.0;
    for(size_t iteration=0;iteration<iterations;iteration++)
    {
        if(iteration%output_period==0)
        {
            print("Iteration:", iteration);
            error/=output_period;
            print("Error:", error);
            error=0.0;
            print("Seconds:", elapsed_seconds()/output_period);
            produce_output(neural_network->get_layers());
        }
        // print("Iteration", iteration, "started...");
        neural_network->iterate();
        error+=neural_network->get_error();
    }

    return 0;
}
