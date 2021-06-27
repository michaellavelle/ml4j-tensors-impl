package org.ml4j.tensor;

import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.AutogradValueCreator;
import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.impl.ValueNodeWrapper;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.DifferentiableBinaryOperator;
import org.ml4j.autograd.operators.DifferentiableUnaryOperator;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.tensor.djl.DJLTensor;
import org.ml4j.tensor.djl.DJLTensorOperations;

import java.util.function.Supplier;

public abstract class TensorWrapperImpl<S extends Tensor<S, D>, T extends Tensor<T, E>, D, E> implements AutogradValue<T, E, Size>, TensorOperations<T>, org.ml4j.autograd.DataSupplier<E>, Tensor<T, E> {

    protected S t;
    protected T cachedGrad;

    public TensorWrapperImpl(S t) {

        this.t = t;
        if (t == null) {
            throw new IllegalArgumentException();
        }
    }

    public S getT() {
        return t;
    }

    @Override
    public float[] getDataAsFloatArray() {
        return t.getDataAsFloatArray();
    }

    @Override
    public T requires_grad_(boolean b) {
        t.requires_grad_(b);
        return get();
    }

    @Override
    public boolean requires_grad() {
        return t.requires_grad();
    }

    @Override
    public T grad() {
        if (cachedGrad != null) {
            return cachedGrad;
        }
        S g = t.grad();
        if (g == null) {
            return null;
        } else {
            cachedGrad = create(g);
            return cachedGrad;
        }
    }

    @Override
    public void backward() {
        System.out.println("CCC:" + t.grad());
        t.backward();
    }

    @Override
    public void backward(BackwardConfig backwardConfig) {
        System.out.println("DDD:" + t.grad());

        t.backward(backwardConfig);
    }

    @Override
    public void backward(T ml4JTensor) {
        t.backward(extract(ml4JTensor));
    }

    @Override
    public void backward(T ml4JTensor, BackwardConfig backwardConfig) {
        t.backward(extract(ml4JTensor), backwardConfig);
    }

    protected abstract S extract(T tensor);

    protected abstract T create(S tensor);

    @Override
    public void swapWith(T ml4JTensor) {
        t.swapWith(extract(ml4JTensor));
    }

    @Override
    public T add(T ml4JTensor) {
        return create(t.add(extract(ml4JTensor)));
    }

    @Override
    public T add(float v) {
        return create(t.add(v));
    }

    @Override
    public T sub(T ml4JTensor) {
        return create(t.sub(extract(ml4JTensor)));
    }

    @Override
    public T sub(float v) {
        return create(t.sub(v));
    }

    @Override
    public T mul(T ml4JTensor)
    {
        return create(t.mul(extract(ml4JTensor)));
    }

    @Override
    public T mul(float v) {
        return create(t.mul(v));
    }

    @Override
    public T div(T ml4JTensor) {
        return create(t.div(extract(ml4JTensor)));
    }

    @Override
    public T div(float v) {
        return create(t.div(v));
    }

    @Override
    public T add_(T ml4JTensor) {
        t.add_(extract(ml4JTensor));
        return get();
    }

    @Override
    public T sub_(T ml4JTensor) {
        t.sub_(extract(ml4JTensor));
        return get();
    }

    @Override
    public T neg() {
        return create(t.neg());

    }

    @Override
    public T gt(float v) {
        return create(t.gt(v));
    }

    @Override
    public T gte(float v) {
        return create(t.gte(v));
    }

    @Override
    public Supplier<E> data() {
        return () -> createData(t.data().get());
    }

    protected abstract E createData(D data);

    @Override
    public T data_(Supplier<E> supplier) {
        t.data_(() -> extractData(supplier.get()));
        return get();
    }

    protected abstract D extractData(E ml4JTensorOperations);

    @Override
    public String name() {
        return t.name();
    }

    @Override
    public T name_(String s)
    {
        t.name_(s);
        return get();
    }

    @Override
    public T self() {
        return get();
    }

    @Override
    public Size context() {
        return t.context();
    }

    @Override
    public boolean create_graph() {
        return t.create_graph();
    }

    @Override
    public int numel() {
        return t.numel();
    }

    @Override
    public T sum(int... dims)
    {
        return create(t.sum(dims));
    }

    @Override
    public float get(int index) {
        return t.get(index);
    }

    @Override
    public float get(int... indexes) {
        return t.get(indexes);
    }

    @Override
    public T getTensor(int... indexes) {
        return create(t.getTensor(indexes));
    }

    @Override
    public void putTensor(T tensor, int... indexes) {
        t.putTensor(extract(tensor), indexes);
    }

    @Override
    public T argMax(int axis) {
        return create(t.argMax(axis));
    }

    @Override
    public T argMax() {
        return create(t.argMax());
    }

    @Override
    public void put(int index, float value) {
        t.put(index, value);
    }

    @Override
    public void put(float value, int... indexes) {
        t.put(value, indexes);
    }

    @Override
    public T mean() {
        return create(t.mean());

    }

    @Override
    public T norm() {
        return create(t.norm());
    }

    @Override
    public T mul_(T other) {
        t.div_(extract(other));
        return get();
    }

    @Override
    public T add_(float v) {
        t.add_(v);
        return get();
    }

    @Override
    public T div_(float v) {
        t.div_(v);
        return get();
    }

    @Override
    public T sub_(float v) {
        t.sub_(v);
        return get();
    }

    @Override
    public T div_(T other) {
        t.div_(extract(other));
        return get();
    }

    @Override
    public T mul_(float v) {
        t.mul_(v);
        return get();
    }

    @Override
    public T columnSums() {
        return create(t.columnSums());
    }

    @Override
    public T rowSums() {
        return create(t.rowSums());
    }

    @Override
    public T cloneTensor() {
        return create(t.cloneTensor());
    }

    @Override
    public T matmul(T other) {
        return create(t.matmul(extract(other)));
    }

    @Override
    public T t() {
        return create(t.t());
    }

    @Override
    public Size size() {
        return t.size();
    }

    @Override
    public int size(int dim) {
        return t.size(dim);
    }

    /*
    @Override
    public T size_(Size size) {
        t.size_(size);
        return get();
    }

     */

    @Override
    public T zero_() {
        t.zero_();
        return get();
    }

    @Override
    public T normal_(float v1, float v2) {
        t.normal_(v1, v2);
        return get();
    }

    @Override
    public T fill_(float value) {
        t.fill_(value);
        return get();
    }

    @Override
    public T view(Size size) {
        return create(t.view(size));
    }

    @Override
    public T resize_(Size size) {
        t.resize_(size);
        return get();
    }

    @Override
    public T reshape(Size size) {
        return create(t.reshape(size));
    }

    @Override
    public T view(int... dims) {
        return create(t.view(dims));
    }

    @Override
    public T relu() {
        return create(t.relu());
    }

    @Override
    public T bernoulli() {
        return create(t.bernoulli());
    }

    @Override
    public T sigmoid() {
        return create(t.sigmoid());
    }

    @Override
    public T exp() {
        return create(t.log());
    }

    @Override
    public T log() {
        return create(t.log());
    }

    @Override
    public void close() {
        t.close();
    }

    @Override
    public boolean isNativeGradient() {
        return t.isNativeGradient();
    }

}
