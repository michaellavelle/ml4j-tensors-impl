package org.ml4j.tensor;

import org.apache.commons.lang3.tuple.Pair;
import org.jvmpy.symbolictensors.MultiplicationRules;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.Value;
import org.ml4j.autograd.arithmetic.operations.DifferentiableWrappedArithmeticOperations;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.Node;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;

public abstract class DifferentiableWrappedTensorOperations<V extends DifferentiableWrappedTensorOperations<V, D> & Value<V, D, Size>, D extends TensorOperations<D>> extends AutogradValueImpl<V, D, Size> implements AutogradValue<V, D, Size>, TensorOperations<V>, org.ml4j.autograd.DataSupplier<D>, Tensor<V, D>, DifferentiableWrappedArithmeticOperations<V, D, Size> {

    public DifferentiableWrappedTensorOperations(Supplier<D> data, Size context, List<Node<?>> children, boolean requires_grad, boolean create_graph) {
        super(data, context, children, requires_grad, create_graph);
    }

    @Override
    public V add(V other) {
        Size broadcast = MultiplicationRules.getBroadcast(size(), other.size());
        return applyBinaryOperator(other, D::add, (g, p) -> g, (g, p) -> g, "add:" + size() + ":" + other.context(), (f, s) -> broadcast);
    }

    @Override
    public V applyBinaryOperator(V other, BinaryOperator<D> forward, BiFunction<V, Pair<V, V>, V> backThis, BiFunction<V, Pair<V, V>, V> backOther, String op, BinaryOperator<Size> contextMapper) {
        if (!size().getDimensions().equals(other.size().getDimensions())) {
            try {
                Size broadcastSize = MultiplicationRules.getBroadcast(size(), other.size());

                if (broadcastSize.getDimensions().equals(size().getDimensions()) || broadcastSize.getDimensions().equals(other.size().getDimensions())) {
                    float scale1 = (float) broadcastSize.numel() / (float) size().numel();
                    float scale2 = (float) broadcastSize.numel() / (float) other.size().numel();
                    return super.applyBinaryOperator(other, forward, (g, p) -> getSub(backThis.apply(g, p), size(), scale1).mul(scale1), (g, p) -> getSub(backOther.apply(g, p), other.size(), scale2).mul(scale2), op, (f, s) -> broadcastSize);
                } else {
                    throw new IllegalStateException();
                }
            } catch (IllegalArgumentException e) {
                // Size cannot be broadcast, perhaps it's matMul.
            }
        }
        return super.applyBinaryOperator(other, forward, backThis, backOther, op, contextMapper);
    }

    protected abstract V getSub(V other, Size size, float scale);

    @Override
    public V relu() {
        return applyUnaryOperator(D::relu, (g, v) -> g.mul(v.gt(0)), "gt", s -> s);
    }

    @Override
    public V view(Size size) {
        return applyUnaryOperator(v -> v.view(size), (g, v) -> g.view(size()), "view", s -> size);
    }

    @Override
    public V norm() {
        return applyUnaryOperator(t -> t.norm(), (g, v) -> backwardNotYetImplemented(), "norm", s -> new Size());
    }

    @Override
    public V sum() {
        return applyUnaryOperator(t -> t.sum(), (g, v) -> v.mul(g), "sum", s -> new Size());
    }

    @Override
    public V mean() {
        return applyUnaryOperator(t -> t.mean(), (g, v) -> { return v.mul(0).add(1).mul(g).div(v.numel()); }, "mean", s -> new Size());
    }

    public V backwardNotYetImplemented() {
        throw new UnsupportedOperationException();
    }

    public V t() {
        return applyUnaryOperator(D::t, (g, v) -> g.t(), "t", s -> s.t());
    }

    @Override
    public V sigmoid() {
        return applyUnaryOperator(D::sigmoid, (g, v) -> g.mul(sigGrad(v.getDataAsFloatArray()[0])), "gt", s -> s);
    }

    @Override
    public V reshape_(Size size) {
        return applyUnaryOperator(f -> f.reshape_(size), (g, v) -> g.reshape_(size()), "reshape", s -> size);
    }

    @Override
    public V mul_(V other) {
        return applyInlineBinaryOperator(other, D::mul_, "mul");
    }

    @Override
    public V matmul(V other) {

        Size[] sizes = MultiplicationRules.matmul(size(), other.size());

        return this.applyBinaryOperator(other, (f, s) -> f.reshape_(sizes[0]).matmul(s.reshape_(sizes[1])), (g, p) -> {
            return g.reshape_(sizes[2]).matmul(p.getRight().reshape_(sizes[1]).t()).reshape_(size());
        }, (g, p) -> {
            return g.reshape_(sizes[2]).t().matmul(p.getLeft().reshape_(sizes[0])).t().reshape_(other.size());
        }, "matmul", (f, s) -> {
            Size result =  sizes[3];
            int[] dims = result.dimensions();
            int [] firstDims = new int[dims.length- 1];
            for (int i = 0; i < firstDims.length; i++) {
                firstDims[i] = dims[i];
            }
            return new Size(new Size(firstDims), new Size(dims[dims.length - 1]));
        });
    }

    @Override
    public Size getMappedContext(Size f, Size s) {
        return MultiplicationRules.getBroadcast(f, s);
    }

    @Override
    public V bernoulli() {
        return applyUnaryOperator(D::bernoulli, (g, v) -> g, "gt", s -> s);
    }

    @Override
    public int numel() {
        return size().numel();
    }

    @Override
    public V columnSums() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public V normal_(float v1, float v2) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public V fill_(float value) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public int size(int dim) {
        if (dim == -1) {
            return size().numel();
        } else {
            return size().getDimensions().get(dim);
        }
    }
    @Override
    public Size size() {
        return context();
    }

    @Override
    public V zero_() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public V view(int... dims) {
        if (dims.length == 1 && dims[0] == -1) {
            return applyUnaryOperator(t -> t.view(-1), (g, v) -> g.view(size()), "view", s -> new Size(s.numel()));
        }
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public V rowSums() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    public float sig(float x) {
        return 1f / (1f + (float)Math.exp(-x));
    }

    public float sigGrad(float x) {
        float s = sig(x);
        return s * ( 1 - s);
    }

    @Override
    public float[] getDataAsFloatArray() {
        return data().get().getDataAsFloatArray();
    }
}
