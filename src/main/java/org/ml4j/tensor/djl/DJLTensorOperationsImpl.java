package org.ml4j.tensor.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import org.jvmpy.symbolictensors.Operation;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.tensor.TensorOperations;
import org.ml4j.tensor.ml4j.ML4JTensorOperations;

import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

public class DJLTensorOperationsImpl implements TensorOperations<DJLTensorOperations>, DJLTensorOperations {

    private NDArray ndArray;
    protected Shape shape;

    protected DJLTensorOperationsImpl(NDArray a) {
        this.ndArray = a;
        this.shape = a.getShape();
    }

    protected DJLTensorOperationsImpl(Shape a) {
        this.ndArray = null;
        this.shape = a;
    }

    public DJLTensorOperationsImpl(NDArray ndArray, boolean requires_grad) {
        this.ndArray = ndArray;
        this.shape = ndArray.getShape();
        if (requires_grad && ndArray.hasGradient()) {
            ndArray.attachGradient();
        }
    }

    @Override
    public DJLTensorOperations reshape_(Size size) {
        Size thisSize = this.getSize(shape);
        if (thisSize.numel() != size.numel()) {
            throw new IllegalArgumentException();
        }
        thisSize = size;
        this.shape = getShape(thisSize);
        this.ndArray.reshape(shape);
        return this;
    }

    @Override
    public void performInlineOperation(Operation<DJLTensorOperations, Size> operation) {
        operation.apply(this);
    }

    @Override
    public DJLTensorOperations performUnaryMappingOperation(Operation<DJLTensorOperations, Size> operation) {
        return operation.apply(this);
    }

    @Override
    public int size(int d) {
        if (d < 0) d = size().getDimensions().size() + d;
        return size().dimensions()[d];
    }

    @Override
    public DJLTensorOperations size_(Size size) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations zero_() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations normal_(float v1, float v2) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations fill_(float value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations view(int... ints) {
        if (ints.length == 1) {
            ints[0] = (int) size().numel();
        }
        return applyUnaryOperation(t -> t.reshape(getShape(new Size(ints))));
    }

    @Override
    public void close() {

    }


    @Override
    public DJLTensorOperations view(Size size) {
        return applyUnaryOperation(t -> t.reshape(getShape(size)));
    }


    public DJLTensorOperationsImpl(NDArray ndArray, Shape shape, boolean requires_grad) {
        this.ndArray = ndArray;
        this.shape = shape;
        if (ndArray == null) {
            throw new IllegalArgumentException();
        }
        if (requires_grad && ndArray.hasGradient()) {
            ndArray.attachGradient();
        }
    }

    @Override
    public String toString() {
        return "" + getNDArray().toFloatArray()[0];
    }

    public Shape getShape() {
        return shape;
    }

    public DJLTensorOperationsImpl(Shape shape, float initialValue) {
        this.ndArray = DJLTensorFactory.getManager().ones(shape).mul(initialValue);
        this.shape = shape;
    }

    public NDArray getNDArray() {
        return ndArray;
    }

    public DJLTensorOperations create(NDArray other, boolean requires_grad) {
        return new DJLTensorOperationsImpl(other, requires_grad);
    }

    public final Supplier<DJLTensorOperations> zero(Shape shape) {
        return () -> create(DJLTensorFactory.getManager().zeros(shape), false);
    }

    public final Supplier<DJLTensorOperations> one(Shape shape) {
        return () -> create(DJLTensorFactory.getManager().ones(shape), false);
    }

    @Override
    public DJLTensorOperations mul(DJLTensorOperations other) {
        return applyBinaryOperation(other, (f, s) -> f.mul(s));
    }

    @Override
    public DJLTensorOperations div(DJLTensorOperations other) {
        return applyBinaryOperation(other, (f, s) -> f.div(s));
    }

    @Override
    public DJLTensorOperations t() {
        System.out.println("T:" + size());
        int[] axes = new int[size().dimensions().length];
        axes[0] = axes.length - 1;
        for (int i = 0; i < axes.length - 1; i++) {
            axes[i + 1] = i;
        }
        DJLTensorOperations result = create(getNDArray().transpose(axes), false);
        System.out.println("T2:" + result.size());

        return result;
    }

    @Override
    public DJLTensorOperations mul(float other) {
        return applyWithFloatOperation(other, (f, s) -> f.mul(s));
    }

    @Override
    public DJLTensorOperations div(float other) {
        return applyWithFloatOperation(other, (f, s) -> f.div(s));
    }

    @Override
    public DJLTensorOperations sub(float other) {
        return applyWithFloatOperation(other, (f, s) -> f.sub(s));
    }

    @Override
    public DJLTensorOperations add(DJLTensorOperations other) {

        //System.out.println("Adding1:" + this.size());
        //System.out.println("Adding2" + other.size());

        return applyBinaryOperation(other, (f, s) -> f.add(s));
    }

    protected DJLTensorOperations applyBinaryOperation(DJLTensorOperations other, BinaryOperator<NDArray> op) {
        return create(op.apply(getNDArray(), other.getNDArray()), false);
    }

    protected DJLTensorOperations applyUnaryOperation(UnaryOperator<NDArray> op) {
        return create(op.apply(getNDArray()), false);
    }

    protected DJLTensorOperations applyWithFloatOperation(float other, BiFunction<NDArray, Float, NDArray> op) {
        return create(op.apply(getNDArray(), other), false);
    }

    @Override
    public DJLTensorOperations add(float other) {
        return applyWithFloatOperation(other, (f, s) -> f.add(s));
    }

    @Override
    public DJLTensorOperations relu() {
        return applyUnaryOperation(n -> n.gt(DJLTensorFactory.getManager().zeros(getNDArray().getShape())).mul(getNDArray()));
    }

    @Override
    public DJLTensorOperations bernoulli() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations sigmoid() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations gt(float value) {
        return create(getNDArray().gt(value), false);
    }

    @Override
    public DJLTensorOperations gte(float value) {
        return create(getNDArray().gte(value), false);
    }


    @Override
    public DJLTensorOperations add_(DJLTensorOperations other) {
        getNDArray().addi(other.getNDArray());
        return this;
    }

    @Override
    public DJLTensorOperations sub_(DJLTensorOperations other) {
        getNDArray().subi(other.getNDArray());
        return this;
    }

    @Override
    public DJLTensorOperations neg() {
        return applyUnaryOperation(n -> n.neg());
    }

    @Override
    public DJLTensorOperations sub(DJLTensorOperations other) {
        return applyBinaryOperation(other, (f, s) -> f.sub(s));
    }

    @Override
    public DJLTensorOperations matmul(DJLTensorOperations other) {
        return applyBinaryOperation(other, (f, s) -> f.matMul(s));
    }

    @Override
    public int numel() {
        return size().numel();
    }

    @Override
    public DJLTensorOperations sum() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations mean() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations norm() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations mul_(DJLTensorOperations other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations columnSums() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations rowSums() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations cloneTensor() {
        return toDJLTensorOperations(ndArray.duplicate());
    }

    private DJLTensorOperations toDJLTensorOperations(NDArray matrix) {
        return new DJLTensorOperationsImpl(matrix);
    }

    private Shape getShape(Size size) {
        long[] dims = new long[size.dimensions().length];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = size.dimensions()[i];
        }
        return new Shape(dims);
    }
    /*
    //@Override
    public DJLTensorOperations filter(Range... ranges) {

        int[] dims = new int[ranges.length];
        int ind = 0;
        int zeroCount = 0;
        for (Range r : ranges) {
            dims[ind] = r.getSize((int) size().decompose().get(ind).numel());
            if (dims[ind] == 0) {
                zeroCount++;
            }
            ind++;
        }
        long[] dims2 = new long[ranges.length - zeroCount];
        int ind2 = 0;
        for (int i = 0; i < dims.length; i++) {
            if (dims[i] != 0) {
                dims2[ind2] = dims[i];
                ind2++;
            }
        }

        // TODO
        // TODO Auto-generated method stub

        //Size newSize = new Size(dims2);
        // TODO
        for (long d : dims2) {
            System.out.println("DIM:" + d);
        }
        for (long d : ndArray.getShape().getShape()) {
            System.out.println("SHAPE:" + d);
        }
        boolean same = true;
        if (dims2.length == ndArray.getShape().getShape().length) {
            for (int i = 0; i < dims2.length; i++) {
                if (dims2[i] != ndArray.getShape().getShape()[i]) {
                    same = false;
                }
            }
        }

        return new DJLTensorOperations(ndArray, ndArray.hasGradient());
    }
     */

    @Override
    public float[] getDataAsFloatArray() {

        Number[] b = this.getNDArray().toArray();
        float[] d = new float[b.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = b[i].floatValue();
        }
        return d;
    }

    @Override
    public Size size() {
        return getSize(ndArray.getShape());
    }

    protected Size getSize(Shape shape) {
        int[] dims = new int[shape.getShape().length];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = (int) shape.getShape()[i];
        }
        return new Size(dims);
    }

    @Override
    public DJLTensorOperations get() {
        return this;
    }
}
