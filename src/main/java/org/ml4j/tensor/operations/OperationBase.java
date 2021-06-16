package org.ml4j.tensor.operations;

import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.tensor.TensorOperations;

public interface OperationBase<V extends TensorOperations<V>, D extends TensorOperations<D>> extends AutogradValue<V, D, Size>, TensorOperations<V>, org.ml4j.autograd.DataSupplier<D> {

    @Override
    default float[] getDataAsFloatArray() {
        return data().get().getDataAsFloatArray();
    }

    default V backwardNotYetImplemented() {
        throw new UnsupportedOperationException();
    }

}
