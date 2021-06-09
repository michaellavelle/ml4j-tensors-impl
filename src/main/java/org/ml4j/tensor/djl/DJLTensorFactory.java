package org.ml4j.tensor.djl;

import ai.djl.ndarray.NDManager;
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
    public DJLTensor create(Supplier<DJLTensorOperations> supplier, Size size) {
        return new DJLTensor(supplier, size, false, false);
    }
}
