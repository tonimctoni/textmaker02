#include "matrix.hpp"


#ifndef __LSTMLAYER__
#define __LSTMLAYER__
template<unsigned long input_size, unsigned long output_size, unsigned long batch_size, unsigned long time_steps>
class LstmLayerBase
{
protected:
    static constexpr unsigned long concat_size=input_size+output_size;
    //LSTM states of inputs+h after passed through weights (synapses) and activation function applied to them.
    std::array<Matrix<batch_size,output_size>,time_steps> state_g;
    std::array<Matrix<batch_size,output_size>,time_steps> state_i;
    std::array<Matrix<batch_size,output_size>,time_steps> state_f;
    std::array<Matrix<batch_size,output_size>,time_steps> state_o;
    std::array<Matrix<batch_size,output_size>,time_steps> state_s;
    //Further internal states
    std::array<Matrix<batch_size,output_size>,time_steps> state_st;
    std::array<Matrix<batch_size,output_size>,time_steps> state_h;
    //Deltas
    std::array<Matrix<batch_size,output_size>,time_steps> delta_h;
    std::array<Matrix<batch_size,output_size>,time_steps> delta_o;
    std::array<Matrix<batch_size,output_size>,time_steps> delta_s;
    std::array<Matrix<batch_size,output_size>,time_steps> delta_i;
    std::array<Matrix<batch_size,output_size>,time_steps> delta_g;
    std::array<Matrix<batch_size,output_size>,time_steps> delta_f;
    //These two deltas are used by the next timestep (backpropagation through time)
    std::array<Matrix<batch_size,output_size>,time_steps> delta_ls;//last s
    std::array<Matrix<batch_size,output_size>,time_steps> delta_lh;//last h

    //LSTM weights
    Matrix<input_size, output_size> weights_xg;
    Matrix<input_size, output_size> weights_xi;
    Matrix<input_size, output_size> weights_xf;
    Matrix<input_size, output_size> weights_xo;
    Matrix<output_size, output_size> weights_hg;
    Matrix<output_size, output_size> weights_hi;
    Matrix<output_size, output_size> weights_hf;
    Matrix<output_size, output_size> weights_ho;
    //LSTM biases
    Matrix<1, output_size> bias_g;
    Matrix<1, output_size> bias_i;
    Matrix<1, output_size> bias_f;
    Matrix<1, output_size> bias_o;
public:
    LstmLayerBase()
    {
        weights_xg.randomize_for_nn(concat_size+1);
        weights_xi.randomize_for_nn(concat_size+1);
        weights_xf.randomize_for_nn(concat_size+1);
        weights_xo.randomize_for_nn(concat_size+1);
        weights_hg.randomize_for_nn(concat_size+1);
        weights_hi.randomize_for_nn(concat_size+1);
        weights_hf.randomize_for_nn(concat_size+1);
        weights_ho.randomize_for_nn(concat_size+1);
        bias_g.randomize_for_nn(concat_size+1);
        bias_i.randomize_for_nn(concat_size+1);
        bias_f.randomize_for_nn(concat_size+1);
        bias_o.randomize_for_nn(concat_size+1);
    }

    void show_guts() const noexcept
    {
        print("LstmLayer", input_size, output_size, batch_size, time_steps);
        print("weights_xg:");
        print(weights_xg);
        print("weights_xi:");
        print(weights_xi);
        print("weights_xf:");
        print(weights_xf);
        print("weights_xo:");
        print(weights_xo);
        print("weights_hg:");
        print(weights_hg);
        print("weights_hi:");
        print(weights_hi);
        print("weights_hf:");
        print(weights_hf);
        print("weights_ho:");
        print(weights_ho);
        print("bias_g:");
        print(bias_g);
        print("bias_i:");
        print(bias_i);
        print("bias_f:");
        print(bias_f);
        print("bias_o:");
        print(bias_o);
    }

    bool has_nan() const noexcept
    {
        return weights_xg.has_nan()
            or weights_xi.has_nan()
            or weights_xf.has_nan()
            or weights_xo.has_nan()
            or weights_hg.has_nan()
            or weights_hi.has_nan()
            or weights_hf.has_nan()
            or weights_ho.has_nan()
            or bias_g.has_nan()
            or bias_i.has_nan()
            or bias_f.has_nan()
            or bias_o.has_nan();
    }

    const Matrix<input_size, output_size>& get_weights_xg() const noexcept {return weights_xg;}
    const Matrix<input_size, output_size>& get_weights_xi() const noexcept {return weights_xi;}
    const Matrix<input_size, output_size>& get_weights_xf() const noexcept {return weights_xf;}
    const Matrix<input_size, output_size>& get_weights_xo() const noexcept {return weights_xo;}
    const Matrix<output_size, output_size>& get_weights_hg() const noexcept {return weights_hg;}
    const Matrix<output_size, output_size>& get_weights_hi() const noexcept {return weights_hi;}
    const Matrix<output_size, output_size>& get_weights_hf() const noexcept {return weights_hf;}
    const Matrix<output_size, output_size>& get_weights_ho() const noexcept {return weights_ho;}

    const Matrix<1, output_size>& get_bias_g() const noexcept {return bias_g;}
    const Matrix<1, output_size>& get_bias_i() const noexcept {return bias_i;}
    const Matrix<1, output_size>& get_bias_f() const noexcept {return bias_f;}
    const Matrix<1, output_size>& get_bias_o() const noexcept {return bias_o;}

    template<unsigned long other_batch_size, unsigned long other_time_steps>
    void set_wb(const LstmLayerBase<input_size, output_size, other_batch_size, other_time_steps> &other)
    {
        weights_xg.set(other.get_weights_xg());
        weights_xi.set(other.get_weights_xi());
        weights_xf.set(other.get_weights_xf());
        weights_xo.set(other.get_weights_xo());
        weights_hg.set(other.get_weights_hg());
        weights_hi.set(other.get_weights_hi());
        weights_hf.set(other.get_weights_hf());
        weights_ho.set(other.get_weights_ho());
        bias_g.set(other.get_bias_g());
        bias_i.set(other.get_bias_i());
        bias_f.set(other.get_bias_f());
        bias_o.set(other.get_bias_o());
    }

    void calc(const Matrix<batch_size,input_size> &X, size_t time_step)
    {
        assert(time_step<time_steps);
        //Calculate states g, i (input gate), f (forget gate), and o (output gate). steps are split for readability
        //f does not need to be calculated in the first round (nothing to forget there). maybe optimize that later.

        //Multiply input with corresponding weights for each state
        state_g[time_step].equals_a_dot_b(X,weights_xg);
        state_i[time_step].equals_a_dot_b(X,weights_xi);
        state_f[time_step].equals_a_dot_b(X,weights_xf);
        state_o[time_step].equals_a_dot_b(X,weights_xo);
        if(time_step!=0)
        {
            //Multiply last h-state with corresponding weights for each state
            state_g[time_step].add_a_dot_b(state_h[time_step-1], weights_hg);
            state_i[time_step].add_a_dot_b(state_h[time_step-1], weights_hi);
            state_f[time_step].add_a_dot_b(state_h[time_step-1], weights_hf);
            state_o[time_step].add_a_dot_b(state_h[time_step-1], weights_ho);
        }
        //Add biases to each state
        state_g[time_step].add_to_each_row(bias_g);
        state_i[time_step].add_to_each_row(bias_i);
        state_f[time_step].add_to_each_row(bias_f);
        state_o[time_step].add_to_each_row(bias_o);
        //Apply activation function to each state
        state_g[time_step].apply_tanh();
        state_i[time_step].apply_sigmoid();
        state_f[time_step].apply_sigmoid();
        state_o[time_step].apply_sigmoid();

        //Calculate s-state. This is the "memory state" which passes information to subsecuent timesteps
        if(time_step!=0)state_s[time_step].equals_a_mul_b_add_c_mul_d(state_g[time_step], state_i[time_step], state_s[time_step-1], state_f[time_step]);
        else state_s[time_step].equals_a_mul_b(state_g[time_step], state_i[time_step]);

        //The "memory state" s needs to have a element-wise tanh function applied to it for further calculations
        state_st[time_step].set(state_s[time_step]);
        state_st[time_step].apply_tanh();

        //Calculate the output of the LSTM (tanh of output of mem-cell times output gate)
        state_h[time_step].equals_a_mul_b(state_st[time_step], state_o[time_step]);
    }

    inline void set_first_delta(const Matrix<batch_size,output_size> &Y, size_t time_step)
    {
        assert(time_step<time_steps);
        delta_h[time_step].equals_a_sub_b(Y,state_h[time_step]);
    }

    void propagate_delta(size_t time_step)
    {
        assert(time_step<time_steps);
        //Add deltas from future timesteps to the h-state delta
        if(time_step<time_steps-1)delta_h[time_step].add(delta_lh[time_step+1]);

        //Get delta of the o-state
        delta_o[time_step].equals_a_mul_b(delta_h[time_step], state_st[time_step]);
        delta_o[time_step].mult_after_func01(state_o[time_step]);
        //Get delta of the s-state
        delta_s[time_step].equals_a_mul_b(delta_h[time_step], state_o[time_step]);
        delta_s[time_step].mult_after_func02(state_st[time_step]);
        if(time_step<time_steps-1)delta_s[time_step].add(delta_ls[time_step+1]);
        //Get delta of the i-state
        delta_i[time_step].equals_a_mul_b(delta_s[time_step], state_g[time_step]);
        delta_i[time_step].mult_after_func01(state_i[time_step]);
        //Get delta of the g-state
        delta_g[time_step].equals_a_mul_b(delta_s[time_step], state_i[time_step]);
        delta_g[time_step].mult_after_func02(state_g[time_step]);

        //Get delta of the f-state and last s-state //f state does not exist in the first round anyways
        //Both deltas are not needed in the first round, so they are not calculated
        if(time_step!=0)
        {
            delta_f[time_step].equals_a_mul_b(delta_s[time_step], state_s[time_step-1]);
            delta_f[time_step].mult_after_func01(state_f[time_step]);
            delta_ls[time_step].equals_a_mul_b(delta_s[time_step], state_f[time_step]);

            delta_lh[time_step].equals_a_dot_bt(delta_i[time_step], weights_hi);
            delta_lh[time_step].add_a_dot_bt(delta_f[time_step], weights_hf);
            delta_lh[time_step].add_a_dot_bt(delta_o[time_step], weights_ho);
            delta_lh[time_step].add_a_dot_bt(delta_g[time_step], weights_hg);
        }
    }

    inline void propagate_delta(Matrix<batch_size,input_size> &X_delta, size_t time_step)
    {
        propagate_delta(time_step);

        X_delta.equals_a_dot_bt(delta_i[time_step], weights_xi);
        X_delta.add_a_dot_bt(delta_f[time_step], weights_xf);
        X_delta.add_a_dot_bt(delta_o[time_step], weights_xo);
        X_delta.add_a_dot_bt(delta_g[time_step], weights_xg);
    }

    inline const Matrix<batch_size,output_size>& get_output(size_t time_step) const noexcept
    {
        return state_h[time_step];
    }

    inline Matrix<batch_size,output_size>& get_delta_output(size_t time_step) noexcept
    {
        return delta_h[time_step];
    }

    inline void update_weights_without_optimizer(const Matrix<batch_size,input_size> &X, size_t time_step, double learning_rate)
    {
        assert(time_step<time_steps);
        weights_xg.add_factor_mul_at_dot_b(learning_rate, X, delta_g[time_step]);
        weights_xi.add_factor_mul_at_dot_b(learning_rate, X, delta_i[time_step]);
        weights_xf.add_factor_mul_at_dot_b(learning_rate, X, delta_f[time_step]);
        weights_xo.add_factor_mul_at_dot_b(learning_rate, X, delta_o[time_step]);
        if(time_step!=0)
        {
            weights_hg.add_factor_mul_at_dot_b(learning_rate, state_h[time_step-1], delta_g[time_step]);
            weights_hi.add_factor_mul_at_dot_b(learning_rate, state_h[time_step-1], delta_i[time_step]);
            weights_hf.add_factor_mul_at_dot_b(learning_rate, state_h[time_step-1], delta_f[time_step]);
            weights_ho.add_factor_mul_at_dot_b(learning_rate, state_h[time_step-1], delta_o[time_step]);
        }
        bias_g.add_factor_mul_each_row_of_a(learning_rate, delta_g[time_step]);
        bias_i.add_factor_mul_each_row_of_a(learning_rate, delta_i[time_step]);
        bias_f.add_factor_mul_each_row_of_a(learning_rate, delta_f[time_step]);
        bias_o.add_factor_mul_each_row_of_a(learning_rate, delta_o[time_step]);
    }
};

template<unsigned long input_size, unsigned long output_size, unsigned long batch_size, unsigned long time_steps>
class LstmLayerRMSProp: public LstmLayerBase<input_size, output_size, batch_size, time_steps>
{
private:
    Matrix<input_size, output_size> ms_weights_xg;
    Matrix<input_size, output_size> ms_weights_xi;
    Matrix<input_size, output_size> ms_weights_xf;
    Matrix<input_size, output_size> ms_weights_xo;
    Matrix<output_size, output_size> ms_weights_hg;
    Matrix<output_size, output_size> ms_weights_hi;
    Matrix<output_size, output_size> ms_weights_hf;
    Matrix<output_size, output_size> ms_weights_ho;
    Matrix<1, output_size> ms_bias_g;
    Matrix<1, output_size> ms_bias_i;
    Matrix<1, output_size> ms_bias_f;
    Matrix<1, output_size> ms_bias_o;
public:
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::state_h;

    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::delta_g;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::delta_i;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::delta_f;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::delta_o;

    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::weights_xg;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::weights_xi;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::weights_xf;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::weights_xo;

    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::weights_hg;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::weights_hi;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::weights_hf;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::weights_ho;

    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::bias_g;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::bias_i;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::bias_f;
    using LstmLayerBase<input_size, output_size, batch_size, time_steps>::bias_o;
    LstmLayerRMSProp()noexcept:LstmLayerBase<input_size, output_size, batch_size, time_steps>()//, ms_weights(1.0), ms_bias(1.0)
    {
    }

    inline void update_weights_with_rmsprop(const Matrix<batch_size,input_size> &X, size_t time_step, const double learning_rate, const double decay) noexcept
    {
        assert(time_step<time_steps);
        update_weights_and_ms_with_rmsprop(X, delta_g[time_step], weights_xg, ms_weights_xg, learning_rate, decay);
        update_weights_and_ms_with_rmsprop(X, delta_i[time_step], weights_xi, ms_weights_xi, learning_rate, decay);
        update_weights_and_ms_with_rmsprop(X, delta_f[time_step], weights_xf, ms_weights_xf, learning_rate, decay);
        update_weights_and_ms_with_rmsprop(X, delta_o[time_step], weights_xo, ms_weights_xo, learning_rate, decay);
        if(time_step!=0)
        {
            update_weights_and_ms_with_rmsprop(state_h[time_step-1], delta_g[time_step], weights_hg, ms_weights_hg, learning_rate, decay);
            update_weights_and_ms_with_rmsprop(state_h[time_step-1], delta_i[time_step], weights_hi, ms_weights_hi, learning_rate, decay);
            update_weights_and_ms_with_rmsprop(state_h[time_step-1], delta_f[time_step], weights_hf, ms_weights_hf, learning_rate, decay);
            update_weights_and_ms_with_rmsprop(state_h[time_step-1], delta_o[time_step], weights_ho, ms_weights_ho, learning_rate, decay);
        }
        update_bias_and_ms_with_rmsprop(delta_g[time_step], bias_g, ms_bias_g, learning_rate, decay);
        update_bias_and_ms_with_rmsprop(delta_i[time_step], bias_i, ms_bias_i, learning_rate, decay);
        update_bias_and_ms_with_rmsprop(delta_f[time_step], bias_f, ms_bias_f, learning_rate, decay);
        update_bias_and_ms_with_rmsprop(delta_o[time_step], bias_o, ms_bias_o, learning_rate, decay);
    }
};
#endif