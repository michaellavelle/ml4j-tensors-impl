package org.ml4j.tensor.ml4j;

import org.jvmpy.symbolictensors.Size;
import org.ml4j.MatrixFactory;
import org.ml4j.autograd.impl.AutogradValueProperties;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.tensor.TensorFactory;

import java.util.function.Supplier;

public class ML4JTensorFactory implements TensorFactory<ML4JTensor, ML4JTensorOperations> {

    public static final MatrixFactory DEFAULT_MATRIX_FACTORY = new JBlasRowMajorMatrixFactory();

    public static final DirectedComponentsContext DEFAULT_DIRECTED_COMPONENTS_CONTEXT = new DirectedComponentsContextImpl(DEFAULT_MATRIX_FACTORY, false);

    @Override
    public ML4JTensorImpl create(Supplier<ML4JTensorOperations> supplier, Size size) {
        return new ML4JTensorImpl(DEFAULT_DIRECTED_COMPONENTS_CONTEXT, supplier, new AutogradValueProperties<Size>().setContext(size));
    }

    @Override
    public ML4JTensorImpl create(float[] data, Size size) {
        return new ML4JTensorImpl(DEFAULT_DIRECTED_COMPONENTS_CONTEXT, () -> new ML4JTensorOperationsImpl(DEFAULT_DIRECTED_COMPONENTS_CONTEXT, DEFAULT_MATRIX_FACTORY.createMatrixFromRowsByRowsArray(size.dimensions()[0], size.dimensions()[1], data), size), new AutogradValueProperties<Size>().setContext(size));
    }

    @Override
    public ML4JTensorImpl create(float[] data) {
        return null;
    }

    @Override
    public ML4JTensorImpl create() {
        return null;
    }

    @Override
    public ML4JTensorImpl ones(Size size) {
        return null;
    }

    @Override
    public ML4JTensorImpl zeros(Size size) {
        return null;
    }

    @Override
    public ML4JTensorImpl randn(Size size) {
        return null;
    }

    @Override
    public ML4JTensorImpl rand(Size size) {
        return null;
    }

    @Override
    public ML4JTensorImpl empty(Size size) {
        return null;
    }
}
