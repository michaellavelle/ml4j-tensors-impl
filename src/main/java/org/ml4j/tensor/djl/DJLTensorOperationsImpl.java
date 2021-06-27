package org.ml4j.tensor.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.jvmpy.symbolictensors.Operation;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.tensor.TensorOperations;
import org.ml4j.tensor.dl4j.DL4JTensorOperations;
import org.ml4j.tensor.ml4j.ML4JTensorOperations;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

public class DJLTensorOperationsImpl implements TensorOperations<DJLTensorOperations>, DJLTensorOperations {

    private NDArray ndArray;
    protected Shape shape;
    protected boolean nativeGradient;

    protected DJLTensorOperationsImpl(NDArray a) {
        this.ndArray = a;
        this.shape = a.getShape();
    }

    public DJLTensorOperationsImpl(ML4JTensorOperations other) {
        this.ndArray = DJLTensorFactory.getManager().create(other.getDataAsFloatArray(), DJLTensorFactory.getShape(other.size()));
        this.shape = DJLTensorFactory.getShape(other.size());
    }

    public DJLTensorOperationsImpl(DL4JTensorOperations other) {
        this.ndArray = DJLTensorFactory.getManager().create(other.getDataAsFloatArray(), DJLTensorFactory.getShape(other.size()));
        this.shape = DJLTensorFactory.getShape(other.size());
    }

    public void setNativeGradient(boolean nativeGradient) {
        this.nativeGradient = nativeGradient;
    }

    public boolean isNativeGradient() {
        return nativeGradient;
    }

    protected DJLTensorOperationsImpl(Shape a) {
        this.ndArray = null;
        this.shape = a;
    }

    public DJLTensorOperationsImpl(NDArray ndArray, boolean requires_grad) {
        this.ndArray = ndArray;
        this.shape = ndArray.getShape();
        if (requires_grad) {
            ndArray.setRequiresGradient(true);
        }
    }

    @Override
    public DJLTensorOperations resize_(Size size) {
        Size thisSize = this.getSize(shape);
        if (thisSize.numel() != size.numel()) {
            throw new IllegalArgumentException();
        }
        this.shape = getShape(size);
        this.ndArray = this.ndArray.reshape(this.shape);
        return this;
    }

    @Override
    public DJLTensorOperations reshape(Size size) {
        return applyUnaryOperation(t -> t.reshape(getShape(size)));
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

    /*
    @Override
    public DJLTensorOperations size_(Size size) {
        throw new UnsupportedOperationException();
    }


     */
    @Override
    public DJLTensorOperations zero_() {
        this.ndArray = DJLTensorFactory.getManager().zeros(getShape(size()), DataType.FLOAT32);
        return this;    }

    @Override
    public DJLTensorOperations normal_(float v1, float v2) {
        this.ndArray = DJLTensorFactory.getManager().randomNormal(v1, v2, getShape(size()), DataType.FLOAT32);
        return this;
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
        if (requires_grad) {
            ndArray.setRequiresGradient(true);
        }
    }

    @Override
    public String toString() {
        return "" + getNDArray().toFloatArray()[0];
    }

    public Shape getShape() {
        return shape;
    }

    public DJLTensorOperationsImpl(Shape shape, float initialValue, boolean requires_grad) {
        this.ndArray = DJLTensorFactory.getManager().ones(shape).mul(initialValue);
        if (requires_grad) {
            this.ndArray.setRequiresGradient(true);
        }
        this.shape = shape;
    }

    public NDArray getNDArray() {
        return ndArray;
    }

    public DJLTensorOperations create(NDArray other, boolean requires_grad) {
        if (requires_grad) {
            other.setRequiresGradient(true);
        }
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
        int[] axes = new int[size().dimensions().length];
        axes[0] = axes.length - 1;
        for (int i = 0; i < axes.length - 1; i++) {
            axes[i + 1] = i;
        }
        DJLTensorOperations result = create(getNDArray().transpose(axes), false);

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
        return applyUnaryOperation(n -> extend(n).relu());
    }


    private NDArrayEx extend(NDArray array) {
        return array.getNDArrayInternal();
    }

    @Override
    public DJLTensorOperations bernoulli() {
        throw new UnsupportedOperationException();
    }

    @Override
    public DJLTensorOperations sigmoid() {
        return applyUnaryOperation(n -> extend(n).sigmoid());
    }

    @Override
    public DJLTensorOperations exp() {
        return applyUnaryOperation(n -> n.exp());
    }

    @Override
    public DJLTensorOperations log() {
        return applyUnaryOperation(n -> n.log());
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
        if (other.size().dimensions().length == 1 && (size().dimensions()[1] == 1 || size().dimensions()[0] == 1)) {
            other = other.reshape(size());
        }
        if (size().dimensions().length == 1 && (other.size().dimensions()[1] == 1 || other.size().dimensions()[0] == 1)) {
            reshape(other.size());
        }
        getNDArray().addi(other.getNDArray());
        return this;
    }

    @Override
    public DJLTensorOperations div_(DJLTensorOperations other) {
        getNDArray().divi(other.getNDArray());
        return this;
    }

    @Override
    public DJLTensorOperations mul_(float v) {
        getNDArray().muli(v);
        return this;
    }

    @Override
    public DJLTensorOperations add_(float v) {
        getNDArray().addi(v);
        return this;
    }

    @Override
    public DJLTensorOperations div_(float v) {
        getNDArray().divi(v);
        return this;
    }

    @Override
    public DJLTensorOperations sub_(float v) {
        getNDArray().subi(v);
        return this;
    }

    @Override
    public DJLTensorOperations sub_(DJLTensorOperations other) {
        if (other.size().dimensions().length == 1 && (size().dimensions()[1] == 1 || size().dimensions()[0] == 1)) {
            other.resize_(size());
        }
        if (size().dimensions().length == 1 && (other.size().dimensions()[1] == 1 || other.size().dimensions()[0] == 1)) {
            resize_(other.size());
        }
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
    public DJLTensorOperations sum(int...axes) {
        return create(getNDArray().sum(axes), false);
    }

    @Override
    public float get(int index) {
        return ndArray.getFloat(getIndexes(size(), index));
    }

    @Override
    public float get(int... indexes) {
        long[] inds = new long[indexes.length];
        for (int i = 0; i < indexes.length; i++) {
            inds[i] = indexes[i];
        }
        return ndArray.getFloat(inds);
    }

    @Override
    public DJLTensorOperations getTensor(int... indexes) {
        List<String> inds = new ArrayList<>();
        for (int i = 0; i < indexes.length; i++) {
            inds.add(indexes[i] == -1 ? ":" : (indexes[i] + ""));
        }
        String r = inds.toString().replace("[","").replace("]", "");
        if (indexes.length == 2 && getNDArray().getShape().getShape().length == 1) {
            return create(getNDArray().reshape(1, getShape().getShape()[0]).get(new NDIndex(r)), false);
        } else {
            return create(getNDArray().get(new NDIndex(r)), false);
        }
    }

    @Override
    public void putTensor(DJLTensorOperations tensor, int... indexes) {
        List<String> inds = new ArrayList<>();
        for (int i = 0; i < indexes.length; i++) {
            inds.add(indexes[i] == -1 ? ":" : (indexes[i] + ""));
        }
        String r = inds.toString().replace("[","").replace("]", "");
        getNDArray().set(new NDIndex(r), tensor.getNDArray());
    }

    @Override
    public DJLTensorOperations argMax() {
        return create(getNDArray().argMax(), false);
    }

    @Override
    public DJLTensorOperations argMax(int i) {
        return create(getNDArray().argMax(i), false);
    }

    private static long[] getIndexes(Size size, int index) {
        if (size.dimensions().length < 2) {
            return new long[] {index};
        } else if (size.dimensions().length == 2) {
            int r = index / size.dimensions()[1];
            return new long[] {r, index - r * size.dimensions()[1]};
        } else {
            throw new UnsupportedOperationException();
        }
    }

    private static int getProduct(Size size, int startDim) {
        int product = 1;
        for (int i = startDim; i < size.dimensions().length; i++) {
            product = product * size.dimensions()[i];
        }
        return product;
    }
    
    

    @Override
    public void put(int index, float value) {
        NDIndex ind = new NDIndex(getIndexes(size(), index));
        ndArray.set(ind, value);
    }

    @Override
    public void put(float value, int...indexes) {
        long[] inds = new long[indexes.length];
        for (int i = 0; i < indexes.length; i++) {
            inds[i] = indexes[i];
        }
        NDIndex ind = new NDIndex(inds);
        ndArray.set(ind, value);
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
        getNDArray().muli(other.getNDArray());
        return this;
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
