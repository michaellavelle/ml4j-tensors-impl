package org.ml4j.tensor.ml4j;

import org.ml4j.tensor.TensorMatrixFactory;

public class ML4JTensorMatrixFactory extends TensorMatrixFactory<ML4JTensor, ML4JTensorOperations> {

    public ML4JTensorMatrixFactory() {
        super(new ML4JTensorFactory());
    }
}
