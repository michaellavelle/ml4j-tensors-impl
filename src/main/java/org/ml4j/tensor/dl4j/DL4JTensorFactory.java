package org.ml4j.tensor.dl4j;

import ai.djl.ndarray.types.Shape;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.tensor.TensorFactory;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;

public class DL4JTensorFactory implements TensorFactory<DL4JTensor, DL4JTensorOperations> {

    @Override
    public DL4JTensor create(Supplier<DL4JTensorOperations> supplier, Size size) {
        return new DL4JTensorImpl(supplier, size, false, false);
    }

    @Override
    public DL4JTensor create(float[] data, Size size) {
        System.out.println("Creating:" + size);
        return new DL4JTensorImpl(() -> new DL4JTensorOperationsImpl(Nd4j.create(data, size.dimensions())), size, false, false);
    }

    public static Shape getShape(Size size) {
        long[] dims = new long[size.dimensions().length];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = size.dimensions()[i];
        }
        return new Shape(dims);
    }

    @Override
    public DL4JTensor create(float[] data) {
        return new DL4JTensorImpl(() -> new DL4JTensorOperationsImpl(Nd4j.create(data, new int[] {})), new Size(), false, false);
    }

    @Override
    public DL4JTensor create() {
        return new DL4JTensorImpl(() -> new DL4JTensorOperationsImpl(Nd4j.create(new int[] {})), new Size(), false, false);
    }

    @Override
    public DL4JTensor ones(Size size) {
        return new DL4JTensorImpl(() -> new DL4JTensorOperationsImpl(Nd4j.ones(size.dimensions())), size, false, false);
    }

    @Override
    public DL4JTensor zeros(Size size) {
        return new DL4JTensorImpl(() -> new DL4JTensorOperationsImpl(Nd4j.zeros(size.dimensions())), size, false, false);
    }

    @Override
    public DL4JTensor randn(Size size) {
        return new DL4JTensorImpl(() -> new DL4JTensorOperationsImpl(Nd4j.randn(size.dimensions())), size, false, false);
    }

    @Override
    public DL4JTensor rand(Size size) {
        return new DL4JTensorImpl(() -> new DL4JTensorOperationsImpl(Nd4j.rand(size.dimensions())), size, false, false);
    }

    @Override
    public DL4JTensor empty(Size size) {
        return new DL4JTensorImpl(() -> new DL4JTensorOperationsImpl(Nd4j.create(size.dimensions())), size, false, false);
    }
}
