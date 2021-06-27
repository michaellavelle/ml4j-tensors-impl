package org.ml4j.tensor.dl4j;

import ai.djl.ndarray.types.Shape;
import org.junit.Assert;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.tensor.AutogradTestBase;
import org.nd4j.linalg.factory.Nd4j;


public class DL4JAutogradTest extends AutogradTestBase<DL4JTensor, DL4JTensorOperations> {

    @Override
    protected DL4JTensorImpl createGradValue(float value, boolean requires_grad) {
        return new DL4JTensorImpl(() -> createData(value, size), size, requires_grad, false);
    }

    @Override
    protected DL4JTensorImpl createGradValue(float value, boolean requires_grad, Size size) {
        return new DL4JTensorImpl(() -> createData(value, size), size, requires_grad, false);
    }

    @Override
    protected DL4JTensorImpl createRandomValue(boolean requires_grad, int... dims) {
        return createGradValue((float) Math.random(), requires_grad, new Size(dims));
    }

    @Override
    protected DL4JTensorImpl createOnesValue(boolean requires_grad, int... dims) {
        return createGradValue(1, requires_grad, new Size(dims));
    }

    @Override
    protected DL4JTensorImpl createGradValue(DL4JTensorOperations value, boolean requires_grad) {
        return new DL4JTensorImpl(() -> value, size, requires_grad, false);
    }

    private Shape getShape(Size size) {
        long[] dims = new long[size.dimensions().length];
        int ind = 0;
        for (int dim : size.dimensions()) {
            dims[ind] = dim;
            ind++;
        }
        return new Shape(dims);
    }

    @Override
    protected void assertEquals(DL4JTensorOperations value1, DL4JTensorOperations value2) {
        float[] m1 = value1.getDataAsFloatArray();
        float[] m2 = value2.getDataAsFloatArray();
        Assert.assertEquals(m1.length, m2.length);
        for (int i = 0; i < m1.length; i++) {
            Assert.assertEquals(m1[i], m2[i], 0.01f);
        }
    }

    protected void assertArrayEqual(float[] actual, float[] expected, float delta) {
        Assert.assertArrayEquals(expected, actual, delta);
    }

    @Override
    protected DL4JTensorOperations add(DL4JTensorOperations value1, DL4JTensorOperations value2) {
        return value1.add(value2);
    }

    @Override
    protected DL4JTensorOperations mul(DL4JTensorOperations value1, float value2) {
        return value1.mul(value2);
    }


    @Override
    protected boolean isNativeGradientSupported() {
        return false;
    }

    @Override
    protected boolean isNativeGradientExpected() {
        return false;
    }

    @Override
    protected DL4JTensorOperations createData(float value) {
        float[] data = new float[size.numel()];
        for (int i = 0; i < data.length; i++) {
            data[i] = value;
        }
        return new DL4JTensorOperationsImpl(Nd4j.create(data, size.dimensions()));
    }

    @Override
    protected DL4JTensorOperations createData(float value, Size size) {
        float[] data = new float[size.numel()];
        for (int i = 0; i < data.length; i++) {
            data[i] = value;
        }
        if (size.dimensions().length == 0) {
            return new DL4JTensorOperationsImpl(Nd4j.scalar(value));
        } else {
            return new DL4JTensorOperationsImpl(Nd4j.create(data, size.dimensions()));
        }
    }
}
