package org.ml4j.tensor.djl;

import org.ml4j.tensor.TensorMatrixFactory;

public class DJLTensorMatrixFactory extends TensorMatrixFactory<DJLTensor, DJLTensorOperations> {

    public DJLTensorMatrixFactory() {
        super(new DJLTensorFactory());
    }
}
