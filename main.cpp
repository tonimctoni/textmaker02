#include "tanh_layer.hpp"
#include "lstm_layer.hpp"
#include "softmax_layer.hpp"
#include <unordered_map>
using namespace std;

// template<unsigned long batch_size, unsigned long time_steps>
// void set_XY(array<Matrix<batch_size, 2>,time_steps> &X, array<Matrix<batch_size,1>,time_steps> &Y)
// {
//     static std::random_device rd;
//     static std::mt19937 gen(rd());
//     static std::uniform_int_distribution<size_t> dst(0,(1<<(time_steps-1))-1);

//     for(size_t batch=0;batch<batch_size;batch++)
//     {
//         size_t num1=dst(gen);
//         size_t num2=dst(gen);
//         size_t num3=num1+num2;

//         for(size_t time_step=0;time_step<time_steps;time_step++)
//         {
//             X[time_step][batch][0]=(num1>>time_step)&1;
//             X[time_step][batch][1]=(num2>>time_step)&1;
//             Y[time_step][batch][0]=(num3>>time_step)&1;
//         }
//     }
// }

// int main()
// {
    // static constexpr unsigned long time_steps=4;
    // static constexpr unsigned long batch_size=1;
    // static constexpr unsigned long iterations=1000000;
    // static constexpr double learning_rate=0.002;
    // static constexpr unsigned long max_show=16;
    // static constexpr unsigned long times_show=2;

    // LstmLayerBase<2,1,batch_size,time_steps> layer1;

    // array<Matrix<batch_size, 2>,time_steps> Xs;
    // array<Matrix<batch_size, 1>,time_steps> Ys;

    // for(size_t iteration=0;iteration<iterations;iteration++)
    // {
    //     set_XY(Xs,Ys);

    //     for(size_t time_step=0;time_step<time_steps;time_step++)
    //     {
    //         layer1.calc(Xs[time_step], time_step);
    //     }

    //     for(size_t time_step=time_steps-1;;time_step--)
    //     {
    //         layer1.set_first_delta(Ys[time_step], time_step);
    //         layer1.propagate_delta(time_step);
    //         if(time_step==0) break;
    //     }

    //     for(size_t time_step=0;time_step<time_steps;time_step++)
    //     {
    //         layer1.update_weights_without_optimizer(Xs[time_step], time_step, learning_rate);
    //     }
    // }

    // for(size_t i=0;i<times_show;i++)
    // {
    //     set_XY(Xs,Ys);
    //     for(size_t time_step=0;time_step<time_steps;time_step++) layer1.calc(Xs[time_step], time_step);

    //     for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
    //         cout << Xs[0][batch][0] << " " << Xs[1][batch][0] << " " << Xs[2][batch][0] << " " << Xs[3][batch][0] << " | ";
    //     cout << endl;
    //     for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
    //         cout << Xs[0][batch][1] << " " << Xs[1][batch][1] << " " << Xs[2][batch][1] << " " << Xs[3][batch][1] << " | ";
    //     cout << endl;
    //     for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
    //         cout << "------- | ";
    //     cout << endl;
    //     for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
    //         cout << (layer1.get_output(0)[batch][0]>.5?1:0) << " "
    //         << (layer1.get_output(1)[batch][0]>.5?1:0) << " "
    //         << (layer1.get_output(2)[batch][0]>.5?1:0) << " "
    //         << (layer1.get_output(3)[batch][0]>.5?1:0) << " | ";
    //     cout << endl;
    //     for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
    //         cout << Ys[0][batch][0] << " " << Ys[1][batch][0] << " " << Ys[2][batch][0] << " " << Ys[3][batch][0] << " | ";
    //     cout << endl;
    //     cout << endl;
    // }

//     static constexpr unsigned long mem_size=10;
//     static constexpr unsigned long time_steps=4;
//     static constexpr unsigned long batch_size=4;
//     static constexpr unsigned long iterations=100000;
//     static constexpr double learning_rate=0.02;
//     static constexpr unsigned long max_show=16;
//     static constexpr unsigned long times_show=2;

//     LstmLayerBase<2,mem_size,batch_size,time_steps> layer1;
//     TanhLayerBase<mem_size,1,batch_size,time_steps> layer2;

//     array<Matrix<batch_size, 2>,time_steps> Xs;
//     array<Matrix<batch_size, 1>,time_steps> Ys;

//     for(size_t iteration=0;iteration<iterations;iteration++)
//     {
//         set_XY(Xs,Ys);

//         for(size_t time_step=0;time_step<time_steps;time_step++)
//         {
//             layer1.calc(Xs[time_step], time_step);
//             layer2.calc(layer1.get_output(time_step), time_step);
//         }

//         for(size_t time_step=time_steps-1;;time_step--)
//         {
//             layer2.set_first_delta(Ys[time_step], time_step);
//             layer2.propagate_delta(layer1.get_delta_output(time_step), time_step);
//             // layer1.set_first_delta(Ys[time_step], time_step);
//             layer1.propagate_delta(time_step);
//             if(time_step==0) break;
//         }

//         for(size_t time_step=0;time_step<time_steps;time_step++)
//         {
//             layer1.update_weights_without_optimizer(Xs[time_step], time_step, learning_rate);
//             layer2.update_weights_without_optimizer(layer1.get_output(time_step), time_step, learning_rate);
//         }
//     }

//     for(size_t i=0;i<times_show;i++)
//     {
//         set_XY(Xs,Ys);
//         for(size_t time_step=0;time_step<time_steps;time_step++)
//         {
//             layer1.calc(Xs[time_step], time_step);
//             layer2.calc(layer1.get_output(time_step), time_step);
//         }

//         for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
//             cout << Xs[0][batch][0] << " " << Xs[1][batch][0] << " " << Xs[2][batch][0] << " " << Xs[3][batch][0] << " | ";
//         cout << endl;
//         for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
//             cout << Xs[0][batch][1] << " " << Xs[1][batch][1] << " " << Xs[2][batch][1] << " " << Xs[3][batch][1] << " | ";
//         cout << endl;
//         for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
//             cout << "------- | ";
//         cout << endl;
//         for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
//             cout << (layer2.get_output(0)[batch][0]>.5?1:0) << " "
//             << (layer2.get_output(1)[batch][0]>.5?1:0) << " "
//             << (layer2.get_output(2)[batch][0]>.5?1:0) << " "
//             << (layer2.get_output(3)[batch][0]>.5?1:0) << " | ";
//         cout << endl;
//         for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
//             cout << Ys[0][batch][0] << " " << Ys[1][batch][0] << " " << Ys[2][batch][0] << " " << Ys[3][batch][0] << " | ";
//         cout << endl;
//         cout << endl;
//     }


//     return 0;
// }


int main()
{
    static constexpr unsigned long time_steps=256;
    static constexpr unsigned long batch_size=100;
    static constexpr unsigned long iterations=20;
    static constexpr double learning_rate=0.02;

    const char *asoiaf_filename="../asoiaf/asoiaf.txt";
    static constexpr unsigned long allowed_char_amount=46;
    static constexpr unsigned long lstm_output_size=100;

    // static constexpr unsigned long text_output_length=4096;

    auto layer1=TanhLayerBase<allowed_char_amount, allowed_char_amount/4, batch_size, time_steps>();
    auto layer2=LstmLayerBase<allowed_char_amount/4, lstm_output_size, batch_size, time_steps>();
    auto layer3=SoftmaxLayerBase<lstm_output_size, allowed_char_amount, batch_size, time_steps>();

    const string index_to_char="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);
    string asoiaf_content;
    read_file_to_string(asoiaf_filename, asoiaf_content);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dst_start(0,asoiaf_content.size()-time_steps-1-1);

    vector<OneHots<batch_size,allowed_char_amount>> Xs(time_steps);
    vector<OneHots<batch_size,allowed_char_amount>> Ys(time_steps);

    auto set_XY=[&](){
        for(size_t batch=0;batch<batch_size;batch++)
        {
            size_t start=dst_start(gen);
            for(size_t time_step=0;time_step<time_steps;time_step++)
            {
                Xs[time_step].set(batch, char_to_index[asoiaf_content[start+time_step]]);
                Ys[time_step].set(batch, char_to_index[asoiaf_content[start+time_step+1]]);
            }
        }
    };

    auto calc=[&](){
        for(size_t time_step=0;time_step<time_steps;time_step++)
        {
            layer1.calc(Xs[time_step].get(), time_step);
            layer2.calc(layer1.get_output(time_step), time_step);
            layer3.calc(layer2.get_output(time_step), time_step);
        }
    };

    auto set_deltas=[&](){
        for(size_t time_step=time_steps-1;;time_step--)
        {
            layer3.set_first_delta_and_propagate_with_cross_enthropy(Ys[time_step].get(), layer2.get_delta_output(time_step), time_step);
            layer2.propagate_delta(layer1.get_delta_output(time_step), time_step);
            layer1.propagate_delta(time_step);
            if(time_step==0) break;
        }
    };

    auto learn=[&](){
        for(size_t time_step=0;time_step<time_steps;time_step++)
        {
            layer1.update_weights_without_optimizer(Xs[time_step].get(), time_step, learning_rate);
            layer2.update_weights_without_optimizer(layer1.get_output(time_step), time_step, learning_rate);
            layer3.update_weights_without_optimizer(layer2.get_output(time_step), time_step, learning_rate);
        }
    };

    for(size_t iteration=0;iteration<iterations;iteration++)
    {
        print("Iteration", iteration, "started...");
        set_XY();
        calc();
        set_deltas();
        learn();
    }

    return 0;
}