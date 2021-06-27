package org.ml4j.tensor.dl4j;

import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.impl.GradNodeWrapper;
import org.ml4j.autograd.impl.ValueNodeWrapper;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.DifferentiableBinaryOperator;
import org.ml4j.autograd.operators.DifferentiableUnaryOperator;
import org.ml4j.tensor.TensorWrapperImpl;
import org.ml4j.tensor.djl.DJLFromDL4JTensorWrapperImpl;
import org.ml4j.tensor.djl.DJLTensor;
import org.ml4j.tensor.djl.DJLTensorOperations;
import org.ml4j.tensor.djl.DJLTensorOperationsImpl;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DL4JFromDJLTensorWrapperImpl extends TensorWrapperImpl<DJLTensor, DL4JTensor, DJLTensorOperations, DL4JTensorOperations> implements DL4JTensor {

    public DL4JFromDJLTensorWrapperImpl(DJLTensor t) {
        super(t);
    }

    @Override
    protected DJLTensor extract(DL4JTensor tensor) {
        if(tensor instanceof DL4JFromDJLTensorWrapperImpl) {
            DL4JFromDJLTensorWrapperImpl oth = (DL4JFromDJLTensorWrapperImpl)tensor;
            return oth.t;
        }
        return new DJLFromDL4JTensorWrapperImpl(tensor);
    }

    @Override
    protected DL4JTensor create(DJLTensor tensor) {
        if(tensor instanceof DJLFromDL4JTensorWrapperImpl) {
            DJLFromDL4JTensorWrapperImpl oth = (DJLFromDL4JTensorWrapperImpl)tensor;
            return oth.getT();
        }
        return new DL4JFromDJLTensorWrapperImpl(tensor);
    }

    @Override
    protected DL4JTensorOperations createData(DJLTensorOperations data) {
        return new DL4JTensorOperationsImpl(data);
    }

    @Override
    protected DJLTensorOperations extractData(DL4JTensorOperations ml4JTensorOperations) {
        return new DJLTensorOperationsImpl(ml4JTensorOperations);
    }

    @Override
    public DL4JTensor get() {
        return this;
    }

    @Override
    public ValueNode<DL4JTensor> getValueNode() {
        return new ValueNodeWrapper<>(t.getValueNode(), f -> new DL4JFromDJLTensorWrapperImpl(f).requires_grad_(false), f -> new DJLFromDL4JTensorWrapperImpl(f).requires_grad_(false));
    }

    @Override
    public GradNode<DL4JTensor> getGradNode() {
        return new GradNodeWrapper<>(t.getGradNode(), f -> f == null ? null : new DL4JFromDJLTensorWrapperImpl(f).requires_grad_(false), f -> f == null ? null : new DJLFromDL4JTensorWrapperImpl(f).requires_grad_(false));
    }

    @Override
    public DL4JTensor apply(DifferentiableUnaryOperator<DL4JTensor, DL4JTensorOperations, Size> differentiableUnaryOperator) {
        DL4JTensor tensor = new DL4JTensorImpl(this).apply(differentiableUnaryOperator);
        return tensor;
    }

    @Override
    public DL4JTensor apply(DifferentiableBinaryOperator<DL4JTensor, DL4JTensorOperations, Size> differentiableBinaryOperator, DL4JTensor ml4JTensor) {
        return new DL4JTensorImpl(this).apply(differentiableBinaryOperator, ml4JTensor);
    }

    @Override
    public INDArray getNDArray() {
        if (size().dimensions().length == 0) {
            return Nd4j.scalar(getDataAsFloatArray()[0]);
        } else {
            return Nd4j.create(getDataAsFloatArray(), size().dimensions());
        }
    }
}
