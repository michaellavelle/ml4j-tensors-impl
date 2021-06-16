package org.ml4j.tensor.ml4j;

import org.jvmpy.symbolictensors.Size;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.tensor.TensorFactory;

import java.util.function.Supplier;

public class ML4JTensorFactory implements TensorFactory<ML4JTensor, ML4JTensorOperations> {

    public static final MatrixFactory DEFAULT_MATRIX_FACTORY = new JBlasRowMajorMatrixFactory();

    public static final DirectedComponentsContext DEFAULT_DIRECTED_COMPONENTS_CONTEXT = new DirectedComponentsContextImpl(DEFAULT_MATRIX_FACTORY, false);

    @Override
    public ML4JTensor create(Supplier<ML4JTensorOperations> supplier, Size size) {
        return new ML4JTensor(DEFAULT_DIRECTED_COMPONENTS_CONTEXT, supplier, size, false, false);
    }

    @Override
    public ML4JTensor create(float[] data, Size size) {
        return null;
    }

    @Override
    public ML4JTensor create(float[] data) {
        return null;
    }

    @Override
    public ML4JTensor create() {
        return null;
    }

    @Override
    public ML4JTensor ones(Size size) {
        return null;
    }

    @Override
    public ML4JTensor zeros(Size size) {
        return null;
    }

    @Override
    public ML4JTensor randn(Size size) {
        return null;
    }

    @Override
    public ML4JTensor rand(Size size) {
        return null;
    }

    @Override
    public ML4JTensor empty(Size size) {
        return null;
    }
}
