#include "tanh_layer.hpp"
#include "lstm_layer.hpp"
using namespace std;



int main()
{
    TanhLayerBase<2,5,4,1> layer1;
    TanhLayerBase<5,1,4,1> layer2;
    auto X=Matrix<4,2>{{0,0},{0,1},{1,0},{1,1}};
    auto Y=Matrix<4,1>{{0,},{1,},{1,},{0,}};
    for(size_t i=0;i<10000000;i++)
    {
        layer1.calc(X,0);
        layer2.calc(layer1.get_output(0),0);

        layer2.set_first_delta(Y,0);
        layer2.propagate_delta(layer1.get_delta_output(0),0);
        layer1.propagate_delta(0);

        layer1.update_weights_without_optimizer(X,0,.01);
        layer2.update_weights_without_optimizer(layer1.get_output(0),0,.01);
    }

    layer1.calc(X,0);
    layer2.calc(layer1.get_output(0),0);
    print(layer2.get_output(0));
    return 0;
}