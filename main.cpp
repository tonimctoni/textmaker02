#if defined(_OPENMP)
    #include "omp_matrix.hpp"
#endif
#include "tanh_layer.hpp"
#include "lstm_layer.hpp"
#include "softmax_layer.hpp"
#include <unordered_map>
#include <memory>
using namespace std;

class NeuralNetwork
{
private:
    static constexpr unsigned long time_steps=250;
    static constexpr unsigned long batch_size=100;
    static constexpr double learning_rate=0.02;
    #define asoiaf_filename "../asoiaf/asoiaf.txt"
    static constexpr unsigned long allowed_char_amount=46;
    static constexpr unsigned long lstm_output_size=200;

    const string index_to_char;
    unordered_map<char, size_t> char_to_index;
    string asoiaf_content;
    std::mt19937 gen;
    std::uniform_int_distribution<size_t> dst_start;

    array<OneHots<batch_size,allowed_char_amount>, time_steps> Xs;
    TanhLayerBase<allowed_char_amount, allowed_char_amount/4, batch_size, time_steps> layer1;
    LstmLayerBase<allowed_char_amount/4, lstm_output_size, batch_size, time_steps> layer2;
    SoftmaxLayerBase<lstm_output_size, allowed_char_amount, batch_size, time_steps> layer3;
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
            layer1.update_weights_without_optimizer(Xs[time_step].get(), time_step, learning_rate);
            layer2.update_weights_without_optimizer(layer1.get_output(time_step), time_step, learning_rate);
            layer3.update_weights_without_optimizer(layer2.get_output(time_step), time_step, learning_rate);
        }
    }
public:
    NeuralNetwork():
    index_to_char("! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz"),asoiaf_content(get_file_content_as_string(asoiaf_filename))
    , gen(random_device()()),dst_start(0,asoiaf_content.size()-time_steps-1-1)
    ,layer1(),layer2(),layer3()
    {
        for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
        assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);
    }

    void iterate()
    {
        set_XY();
        calc();
        set_deltas();
        learn();
    }
};

int main()
{
    static constexpr unsigned long iterations=1;
    unique_ptr<NeuralNetwork> neural_network(new NeuralNetwork);

    for(size_t iteration=0;iteration<iterations;iteration++)
    {
        print("Iteration", iteration, "started...");
        neural_network->iterate();
    }

    return 0;
}
// make && time OMP_NUM_THREADS=1 ./a.out