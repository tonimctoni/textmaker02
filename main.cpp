#include "tanh_layer.hpp"
#include "lstm_layer.hpp"
using namespace std;

template<unsigned long batch_size, unsigned long time_steps>
void set_XY(array<Matrix<batch_size, 2>,time_steps> &X, array<Matrix<batch_size,1>,time_steps> &Y)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<size_t> dst(0,(1<<(time_steps-1))-1);

    for(size_t batch=0;batch<batch_size;batch++)
    {
        size_t num1=dst(gen);
        size_t num2=dst(gen);
        size_t num3=num1+num2;

        for(size_t time_step=0;time_step<time_steps;time_step++)
        {
            X[time_step][batch][0]=(num1>>time_step)&1;
            X[time_step][batch][1]=(num2>>time_step)&1;
            Y[time_step][batch][0]=(num3>>time_step)&1;
        }
    }
}

int main()
{
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

    static constexpr unsigned long mem_size=10;
    static constexpr unsigned long time_steps=4;
    static constexpr unsigned long batch_size=4;
    static constexpr unsigned long iterations=100000;
    static constexpr double learning_rate=0.02;
    static constexpr unsigned long max_show=16;
    static constexpr unsigned long times_show=2;

    LstmLayerBase<2,mem_size,batch_size,time_steps> layer1;
    TanhLayerBase<mem_size,1,batch_size,time_steps> layer2;

    array<Matrix<batch_size, 2>,time_steps> Xs;
    array<Matrix<batch_size, 1>,time_steps> Ys;

    for(size_t iteration=0;iteration<iterations;iteration++)
    {
        set_XY(Xs,Ys);

        for(size_t time_step=0;time_step<time_steps;time_step++)
        {
            layer1.calc(Xs[time_step], time_step);
            layer2.calc(layer1.get_output(time_step), time_step);
        }

        for(size_t time_step=time_steps-1;;time_step--)
        {
            layer2.set_first_delta(Ys[time_step], time_step);
            layer2.propagate_delta(layer1.get_delta_output(time_step), time_step);
            // layer1.set_first_delta(Ys[time_step], time_step);
            layer1.propagate_delta(time_step);
            if(time_step==0) break;
        }

        for(size_t time_step=0;time_step<time_steps;time_step++)
        {
            layer1.update_weights_without_optimizer(Xs[time_step], time_step, learning_rate);
            layer2.update_weights_without_optimizer(layer1.get_output(time_step), time_step, learning_rate);
        }
    }

    for(size_t i=0;i<times_show;i++)
    {
        set_XY(Xs,Ys);
        for(size_t time_step=0;time_step<time_steps;time_step++)
        {
            layer1.calc(Xs[time_step], time_step);
            layer2.calc(layer1.get_output(time_step), time_step);
        }

        for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
            cout << Xs[0][batch][0] << " " << Xs[1][batch][0] << " " << Xs[2][batch][0] << " " << Xs[3][batch][0] << " | ";
        cout << endl;
        for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
            cout << Xs[0][batch][1] << " " << Xs[1][batch][1] << " " << Xs[2][batch][1] << " " << Xs[3][batch][1] << " | ";
        cout << endl;
        for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
            cout << "------- | ";
        cout << endl;
        for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
            cout << (layer2.get_output(0)[batch][0]>.5?1:0) << " "
            << (layer2.get_output(1)[batch][0]>.5?1:0) << " "
            << (layer2.get_output(2)[batch][0]>.5?1:0) << " "
            << (layer2.get_output(3)[batch][0]>.5?1:0) << " | ";
        cout << endl;
        for(size_t batch=0;batch<(batch_size<max_show?batch_size:max_show);batch++)
            cout << Ys[0][batch][0] << " " << Ys[1][batch][0] << " " << Ys[2][batch][0] << " " << Ys[3][batch][0] << " | ";
        cout << endl;
        cout << endl;
    }


    return 0;
}