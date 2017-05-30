#include "matrix.hpp"


#ifndef __LSTMLAYER__
#define __LSTMLAYER__
template<unsigned long input_size, unsigned long output_size, unsigned long batch_size, unsigned long time_steps>
class LstmLayerBase
{
private:
    static constexpr unsigned long concat_size=input_size+output_size;
    //LSTM states of inputs+h after passed through weights (synapses) and activation function applied to them.
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> state_g;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> state_i;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> state_f;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> state_o;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> state_s;
    //Further internal states
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> state_st;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> state_h;
    //Deltas
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> delta_h;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> delta_o;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> delta_s;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> delta_i;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> delta_g;
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> delta_f;
    //These two deltas are used by the next timestep (backpropagation through time)
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> delta_ls;//last s
    std::unique_ptr<std::array<Matrix<batch_size,output_size>,time_steps>> delta_lh;//last h

    //LSTM weights
    std::unique_ptr<Matrix<input_size, output_size>> weights_xg;
    std::unique_ptr<Matrix<input_size, output_size>> weights_xi;
    std::unique_ptr<Matrix<input_size, output_size>> weights_xf;
    std::unique_ptr<Matrix<input_size, output_size>> weights_xo;
    std::unique_ptr<Matrix<output_size, output_size>> weights_hg;
    std::unique_ptr<Matrix<output_size, output_size>> weights_hi;
    std::unique_ptr<Matrix<output_size, output_size>> weights_hf;
    std::unique_ptr<Matrix<output_size, output_size>> weights_ho;
    //LSTM biases
    std::unique_ptr<Matrix<1, output_size>> bias_g;
    std::unique_ptr<Matrix<1, output_size>> bias_i;
    std::unique_ptr<Matrix<1, output_size>> bias_f;
    std::unique_ptr<Matrix<1, output_size>> bias_o;
public:
    LstmLayerBase():
    state_g(new std::array<Matrix<batch_size,output_size>,time_steps>),
    state_i(new std::array<Matrix<batch_size,output_size>,time_steps>),
    state_f(new std::array<Matrix<batch_size,output_size>,time_steps>),
    state_o(new std::array<Matrix<batch_size,output_size>,time_steps>),
    state_s(new std::array<Matrix<batch_size,output_size>,time_steps>),
    state_st(new std::array<Matrix<batch_size,output_size>,time_steps>),
    state_h(new std::array<Matrix<batch_size,output_size>,time_steps>),
    delta_h(new std::array<Matrix<batch_size,output_size>,time_steps>),
    delta_o(new std::array<Matrix<batch_size,output_size>,time_steps>),
    delta_s(new std::array<Matrix<batch_size,output_size>,time_steps>),
    delta_i(new std::array<Matrix<batch_size,output_size>,time_steps>),
    delta_g(new std::array<Matrix<batch_size,output_size>,time_steps>),
    delta_f(new std::array<Matrix<batch_size,output_size>,time_steps>),
    delta_ls(new std::array<Matrix<batch_size,output_size>,time_steps>),
    delta_lh(new std::array<Matrix<batch_size,output_size>,time_steps>),
    weights_xg(new Matrix<input_size, output_size>),
    weights_xi(new Matrix<input_size, output_size>),
    weights_xf(new Matrix<input_size, output_size>),
    weights_xo(new Matrix<input_size, output_size>),
    weights_hg(new Matrix<output_size, output_size>),
    weights_hi(new Matrix<output_size, output_size>),
    weights_hf(new Matrix<output_size, output_size>),
    weights_ho(new Matrix<output_size, output_size>),
    bias_g(new Matrix<1, output_size>),
    bias_i(new Matrix<1, output_size>),
    bias_f(new Matrix<1, output_size>),
    bias_o(new Matrix<1, output_size>)
    {
        weights_xg->randomize_for_nn(concat_size+1);
        weights_xi->randomize_for_nn(concat_size+1);
        weights_xf->randomize_for_nn(concat_size+1);
        weights_xo->randomize_for_nn(concat_size+1);
        weights_hg->randomize_for_nn(concat_size+1);
        weights_hi->randomize_for_nn(concat_size+1);
        weights_hf->randomize_for_nn(concat_size+1);
        weights_ho->randomize_for_nn(concat_size+1);
        bias_g->randomize_for_nn(concat_size+1);
        bias_i->randomize_for_nn(concat_size+1);
        bias_f->randomize_for_nn(concat_size+1);
        bias_o->randomize_for_nn(concat_size+1);
    }

    void calc(const Matrix<batch_size,input_size> &X, size_t time_step)
    {
        assert(time_step<time_steps);
        //Calculate states g, i (input gate), f (forget gate), and o (output gate). steps are split for readability
        //f does not need to be calculated in the first round (nothing to forget there). maybe optimize that later.

        //Multiply input with corresponding weights for each state
        (*state_g)[time_step].equals_a_dot_b(X,*weights_xg);
        (*state_i)[time_step].equals_a_dot_b(X,*weights_xi);
        (*state_f)[time_step].equals_a_dot_b(X,*weights_xf);
        (*state_o)[time_step].equals_a_dot_b(X,*weights_xo);
        if(time_step!=0)
        {
            //Multiply last h-state with corresponding weights for each state
            (*state_g)[time_step].add_a_dot_b((*state_h)[time_step-1], *weights_hg);
            (*state_i)[time_step].add_a_dot_b((*state_h)[time_step-1], *weights_hi);
            (*state_f)[time_step].add_a_dot_b((*state_h)[time_step-1], *weights_hf);
            (*state_o)[time_step].add_a_dot_b((*state_h)[time_step-1], *weights_ho);
        }
        //Add biases to each state
        (*state_g)[time_step].add_to_each_row(*bias_g);
        (*state_i)[time_step].add_to_each_row(*bias_i);
        (*state_f)[time_step].add_to_each_row(*bias_f);
        (*state_o)[time_step].add_to_each_row(*bias_o);
        //Apply activation function to each state
        (*state_g)[time_step].apply_tanh();
        (*state_i)[time_step].apply_sigmoid();
        (*state_f)[time_step].apply_sigmoid();
        (*state_o)[time_step].apply_sigmoid();

        //Calculate s-state. This is the "memory state" which passes information to subsecuent timesteps
        if(time_step!=0)(*state_s)[time_step].equals_a_mul_b_add_c_mul_d((*state_g)[time_step], (*state_i)[time_step], (*state_s)[time_step-1], (*state_f)[time_step]);
        else (*state_s)[time_step].equals_a_mul_b((*state_g)[time_step], (*state_i)[time_step]);

        //The "memory state" s needs to have a element-wise tanh function applied to it for further calculations
        (*state_st)[time_step].set((*state_s)[time_step]);
        (*state_st)[time_step].apply_tanh();

        //Calculate the output of the LSTM (tanh of output of mem-cell times output gate)
        (*state_h)[time_step].equals_a_mul_b((*state_st)[time_step], (*state_o)[time_step]);
    }

    inline void set_first_delta(const Matrix<batch_size,output_size> &Y, size_t time_step)
    {
        assert(time_step<time_steps);
        (*delta_h)[time_step].equals_a_sub_b(Y,(*state_h)[time_step]);
    }

    void propagate_delta(size_t time_step)
    {
        assert(time_step<time_steps);
        //Add deltas from future timesteps to the h-state delta
        if(time_step<time_steps-1)(*delta_h)[time_step].add((*delta_lh)[time_step+1]);

        //Get delta of the o-state
        (*delta_o)[time_step].equals_a_mul_b((*delta_h)[time_step], (*state_st)[time_step]);
        (*delta_o)[time_step].mult_after_func01((*state_o)[time_step]);
        //Get delta of the s-state
        (*delta_s)[time_step].equals_a_mul_b((*delta_h)[time_step], (*state_o)[time_step]);
        (*delta_s)[time_step].mult_after_func02((*state_st)[time_step]);
        if(time_step<time_steps-1)(*delta_s)[time_step].add((*delta_ls)[time_step+1]);
        //Get delta of the i-state
        (*delta_i)[time_step].equals_a_mul_b((*delta_s)[time_step], (*state_g)[time_step]);
        (*delta_i)[time_step].mult_after_func01((*state_i)[time_step]);
        //Get delta of the g-state
        (*delta_g)[time_step].equals_a_mul_b((*delta_s)[time_step], (*state_i)[time_step]);
        (*delta_g)[time_step].mult_after_func02((*state_g)[time_step]);

        //Get delta of the f-state and last s-state //f state does not exist in the first round anyways
        //Both deltas are not needed in the first round, so they are not calculated
        if(time_step!=0)
        {
            (*delta_f)[time_step].equals_a_mul_b((*delta_s)[time_step], (*state_s)[time_step-1]);
            (*delta_f)[time_step].mult_after_func01((*state_f)[time_step]);
            (*delta_ls)[time_step].equals_a_mul_b((*delta_s)[time_step], (*state_f)[time_step]);

            (*delta_lh)[time_step].equals_a_dot_bt((*delta_i)[time_step], *weights_hi);
            (*delta_lh)[time_step].add_a_dot_bt((*delta_f)[time_step], *weights_hf);
            (*delta_lh)[time_step].add_a_dot_bt((*delta_o)[time_step], *weights_ho);
            (*delta_lh)[time_step].add_a_dot_bt((*delta_g)[time_step], *weights_hg);
        }
    }

    inline void propagate_delta(Matrix<batch_size,input_size> &X_delta, size_t time_step)
    {
        propagate_delta(time_step);

        X_delta.equals_a_dot_bt((*delta_i)[time_step], *weights_xi);
        X_delta.add_a_dot_bt((*delta_f)[time_step], *weights_xf);
        X_delta.add_a_dot_bt((*delta_o)[time_step], *weights_xo);
        X_delta.add_a_dot_bt((*delta_g)[time_step], *weights_xg);
    }

    inline const Matrix<batch_size,output_size>& get_output(size_t time_step) const noexcept
    {
        return (*state_h)[time_step];
    }

    inline Matrix<batch_size,output_size>& get_delta_output(size_t time_step) noexcept
    {
        return (*delta_h)[time_step];
    }

    inline void update_weights_without_optimizer(const Matrix<batch_size,input_size> &X, size_t time_step, double learning_rate)
    {
        assert(time_step<time_steps);
        (*weights_xg).add_factor_mul_at_dot_b(learning_rate, X, (*delta_g)[time_step]);
        (*weights_xi).add_factor_mul_at_dot_b(learning_rate, X, (*delta_i)[time_step]);
        (*weights_xf).add_factor_mul_at_dot_b(learning_rate, X, (*delta_f)[time_step]);
        (*weights_xo).add_factor_mul_at_dot_b(learning_rate, X, (*delta_o)[time_step]);
        if(time_step!=0)
        {
            (*weights_hg).add_factor_mul_at_dot_b(learning_rate, (*state_h)[time_step-1], (*delta_g)[time_step]);
            (*weights_hi).add_factor_mul_at_dot_b(learning_rate, (*state_h)[time_step-1], (*delta_i)[time_step]);
            (*weights_hf).add_factor_mul_at_dot_b(learning_rate, (*state_h)[time_step-1], (*delta_f)[time_step]);
            (*weights_ho).add_factor_mul_at_dot_b(learning_rate, (*state_h)[time_step-1], (*delta_o)[time_step]);
        }
        (*bias_g).add_each_row_of_a((*delta_g)[time_step]);
        (*bias_i).add_each_row_of_a((*delta_i)[time_step]);
        (*bias_f).add_each_row_of_a((*delta_f)[time_step]);
        (*bias_o).add_each_row_of_a((*delta_o)[time_step]);
    }
};
#endif