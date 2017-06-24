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
using namespace std;

template<typename T>
void get_type(T t);

class NeuralNetwork
{
private:
    static constexpr unsigned long time_steps=250;
    static constexpr unsigned long batch_size=23;
    static constexpr double learning_rate=0.02;
    static constexpr double decay=0.9;
    #define asoiaf_filename "../asoiaf/asoiaf.txt"
    #define allowed_chars "! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz"
    static constexpr unsigned long allowed_char_amount=46;
    static constexpr unsigned long lstm_output_size=503;

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
            // assert(!layer1.has_nan());
            // assert(!layer1.get_output(time_step).has_nan());
            // assert(!layer1.get_delta_output(time_step).has_nan());
            layer2.calc(layer1.get_output(time_step), time_step);
            // assert(!layer2.has_nan());
            // assert(!layer2.get_output(time_step).has_nan());
            // assert(!layer2.get_delta_output(time_step).has_nan());
            layer3.calc(layer2.get_output(time_step), time_step);
            // assert(!layer3.has_nan());
            // assert(!layer3.get_output(time_step).has_nan());
            // assert(!layer3.get_delta_output(time_step).has_nan());
        }
    }

    void set_deltas()
    {
        for(size_t time_step=time_steps-1;;time_step--)
        {
            layer3.set_first_delta_and_propagate_with_cross_enthropy(Ys[time_step].get(), layer2.get_delta_output(time_step), time_step);
            // assert(!layer3.has_nan());
            // assert(!layer3.get_output(time_step).has_nan());
            // assert(!layer3.get_delta_output(time_step).has_nan());
            layer2.propagate_delta(layer1.get_delta_output(time_step), time_step);
            // assert(!layer2.has_nan());
            // assert(!layer2.get_output(time_step).has_nan());
            // assert(!layer2.get_delta_output(time_step).has_nan());
            layer1.propagate_delta(time_step);
            // assert(!layer1.has_nan());
            // assert(!layer1.get_output(time_step).has_nan());
            // assert(!layer1.get_delta_output(time_step).has_nan());
            if(time_step==0) break;
        }
    }

    void learn()
    {
        for(size_t time_step=0;time_step<time_steps;time_step++)
        {
            // layer1.update_weights_without_optimizer(Xs[time_step].get(), time_step, learning_rate);
            // layer2.update_weights_without_optimizer(layer1.get_output(time_step), time_step, learning_rate);
            // layer3.update_weights_without_optimizer(layer2.get_output(time_step), time_step, learning_rate);
            layer1.update_weights_with_rmsprop(Xs[time_step].get(), time_step, learning_rate, decay);
            // assert(!layer1.has_nan());
            // assert(!layer1.get_output(time_step).has_nan());
            // assert(!layer1.get_delta_output(time_step).has_nan());
            layer2.update_weights_with_rmsprop(layer1.get_output(time_step), time_step, learning_rate, decay);
            // assert(!layer2.has_nan());
            // assert(!layer2.get_output(time_step).has_nan());
            // assert(!layer2.get_delta_output(time_step).has_nan());
            layer3.update_weights_with_rmsprop(layer2.get_output(time_step), time_step, learning_rate, decay);
            // assert(!layer3.has_nan());
            // assert(!layer3.get_output(time_step).has_nan());
            // assert(!layer3.get_delta_output(time_step).has_nan());
        }
        layer3.normalize01();
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

    // NeuralNetwork(UninitializedNeuralNetwork _)
    // {}

    // tuple<const TanhLayerRMSProp<allowed_char_amount, allowed_char_amount/4, batch_size, time_steps>&, 
    // const LstmLayerRMSProp<allowed_char_amount/4, lstm_output_size, batch_size, time_steps>&,
    // const SoftmaxLayerRMSProp<lstm_output_size, allowed_char_amount, batch_size, time_steps>&
    // > get_layers() const
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

    void iterate()
    {
        set_XY();
        calc();
        set_deltas();
        learn();
    }
};

// class NeuralNetwokrOutputProducer
// {
// private:
//     // static constexpr unsigned long time_steps=250;
//     // static constexpr unsigned long batch_size=100;
//     static constexpr unsigned long allowed_char_amount=46;
//     static constexpr unsigned long lstm_output_size=250;
//     TanhLayerBase<allowed_char_amount, allowed_char_amount/4, 1, 1024> layer1;
//     LstmLayerBase<allowed_char_amount/4, lstm_output_size, 1, 1024> layer2;
//     SoftmaxLayerBase<lstm_output_size, allowed_char_amount, 1, 1024> layer3;
// public:
//     // template <unsigned long batch_size, unsigned long time_steps>
//     // NeuralNetwokrOutputProducer(const NeuralNetwork &nn)
//     NeuralNetwokrOutputProducer(tuple<TanhLayerRMSProp<46ul, 11ul, 100ul, 250ul> const&, LstmLayerRMSProp<11ul, 250ul, 100ul, 250ul> const&, SoftmaxLayerRMSProp<250ul, 46ul, 100ul, 250ul> const&> t)
//     {
//         layer1.set_wb(get<0>(t));
//         layer2.set_wb(get<1>(t));
//         layer3.set_wb(get<2>(t));
//     }
// };

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
        // try
        // {
            size_t new_char_index=get_weighted_random_index(layer3->get_output(time_step)[0]);
            X.set(0, new_char_index);
            cout << index_to_char[new_char_index];
        // }
        // catch(...)
        // {
        //     print("Caught");
        //     break;
        // }
    }
    cout << "\n\n" << endl;
}



// class NeuralNetwork
// {
// private:
//     static constexpr double learning_rate=0.02;
//     static constexpr double decay=0.9;
//     Matrix<4,2> X;
//     TanhLayerRMSProp<2, 3, 4, 1> layer1;
//     TanhLayerRMSProp<3, 1, 4, 1> layer2;
//     Matrix<4,1> Y;

//     void calc()
//     {
//         layer1.calc(X, 0);
//         layer2.calc(layer1.get_output(0), 0);
//     }

//     void set_deltas()
//     {
//         layer2.set_first_delta(Y,0);
//         layer2.propagate_delta(layer1.get_delta_output(0),0);
//         layer1.propagate_delta(0);
//     }

//     void learn()
//     {
//         layer1.update_weights_with_rmsprop(X, 0, learning_rate, decay);
//         layer2.update_weights_with_rmsprop(layer1.get_output(0), 0, learning_rate, decay);
//     }
// public:
//     NeuralNetwork():X{{0,0},{0,1},{1,0},{1,1}}, layer1(), Y{{0},{1},{1},{0}}
//     {
//     }

//     void iterate()
//     {
//         // set_XY();
//         calc();
//         set_deltas();
//         learn();
//     }

//     void show()
//     {
//         calc();
//         print(layer2.get_output(0));
//     }
// };

double elapsed_seconds()
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
    unique_ptr<NeuralNetwork> neural_network(new NeuralNetwork);
    elapsed_seconds();
    // neural_network->show_guts();
    for(size_t iteration=0;iteration<iterations;iteration++)
    {
        print("Iteration", iteration, "started...");
        neural_network->iterate();
        // neural_network->show_guts();
        if((iteration+1)%20==0)
        {
            print("Seconds:", elapsed_seconds());
            produce_output(neural_network->get_layers());
        }
    }

    return 0;
}
