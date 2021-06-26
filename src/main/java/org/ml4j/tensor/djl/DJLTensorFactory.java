package org.ml4j.tensor.djl;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtEngine;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.tensor.TensorFactory;

import java.util.function.Supplier;

public class DJLTensorFactory implements TensorFactory<DJLTensor, DJLTensorOperations> {

    static NDManager manager = PtEngine.getInstance().newBaseManager();

    public static NDManager getManager() {
        return manager;
    }

    @Override
    public DJLTensorImpl create(Supplier<DJLTensorOperations> supplier, Size size) {
        return new DJLTensorImpl(supplier, size, false, false);
    }

    @Override
    public DJLTensorImpl create(float[] data, Size size) {
        System.out.println("Creating:" + size);
        return new DJLTensorImpl(() -> new DJLTensorOperationsImpl(manager.create(data, getShape(size))), size, false, false);
    }

    public static Shape getShape(Size size) {
        long[] dims = new long[size.dimensions().length];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = size.dimensions()[i];
        }
        return new Shape(dims);
    }

    @Override
    public DJLTensorImpl create(float[] data) {
        return new DJLTensorImpl(() -> new DJLTensorOperationsImpl(manager.create(data, new Shape())), new Size(), false, false);
    }

    @Override
    public DJLTensorImpl create() {
        return new DJLTensorImpl(() -> new DJLTensorOperationsImpl(manager.create(new Shape())), new Size(), false, false);
    }

    @Override
    public DJLTensorImpl ones(Size size) {
        return new DJLTensorImpl(() -> new DJLTensorOperationsImpl(manager.ones(getShape(size))), size, false, false);
    }

    @Override
    public DJLTensorImpl zeros(Size size) {
        return new DJLTensorImpl(() -> new DJLTensorOperationsImpl(manager.zeros(getShape(size))), size, false, false);
    }

    @Override
    public DJLTensorImpl randn(Size size) {
        return new DJLTensorImpl(() -> new DJLTensorOperationsImpl(manager.randomNormal(getShape(size))), size, false, false);
    }

    @Override
    public DJLTensorImpl rand(Size size) {
        return new DJLTensorImpl(() -> new DJLTensorOperationsImpl(manager.randomUniform(0, 1, getShape(size))), size, false, false);
    }

    @Override
    public DJLTensorImpl empty(Size size) {
        return new DJLTensorImpl(() -> new DJLTensorOperationsImpl(manager.create(getShape(size))), size, false, false);
    }
}
