package org.ml4j.tensor;

import org.apache.commons.lang3.tuple.Pair;
import org.jvmpy.symbolictensors.MultiplicationRules;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.Value;
import org.ml4j.autograd.arithmetic.operations.ArithmeticOperations;
import org.ml4j.autograd.arithmetic.operations.DifferentiableWrappedArithmeticOperations;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.Node;
import org.ml4j.tensor.ml4j.ML4JTensor;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;

public abstract class TensorBase<V extends TensorBase<V, D> & Value<V, D, Size>, D extends TensorOperations<D>> extends AutogradValueImpl<V, D, Size> implements AutogradValue<V, D, Size>, DifferentiableWrappedTensorOperations<V, D>, TensorOperations<V>, org.ml4j.autograd.DataSupplier<D>, Tensor<V, D> {

    public TensorBase(Supplier<D> data, Size context, List<Node<?>> children, boolean requires_grad, boolean create_graph) {
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
}
