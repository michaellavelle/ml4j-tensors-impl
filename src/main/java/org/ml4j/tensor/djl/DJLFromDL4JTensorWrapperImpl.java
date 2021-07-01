package org.ml4j.tensor.djl;

import ai.djl.pytorch.engine.PtNDArray;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.impl.GradNodeWrapper;
import org.ml4j.autograd.impl.ValueNodeWrapper;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.DifferentiableBinaryOperator;
import org.ml4j.autograd.operators.DifferentiableUnaryOperator;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.tensor.TensorWrapperImpl;
import org.ml4j.tensor.dl4j.*;
import org.ml4j.tensor.ml4j.ML4JTensor;


public class DJLFromDL4JTensorWrapperImpl extends TensorWrapperImpl<DL4JTensor, DJLTensor, DL4JTensorOperations, DJLTensorOperations> implements DJLTensor {

    private GradNode<DJLTensor> gradNode;
    private ValueNode<DJLTensor> valueNode;

    @Override
    public void backward() {
        super.backward();
    }

    public DJLFromDL4JTensorWrapperImpl(DL4JTensor t) {
        super(t);
        this.gradNode =  new GradNodeWrapper<>(t.getGradNode(), f -> new DJLFromDL4JTensorWrapperImpl(f), f -> new DL4JFromDJLTensorWrapperImpl(f));
        this.valueNode =  new ValueNodeWrapper<>(t.getValueNode(), f -> new DJLFromDL4JTensorWrapperImpl(f), f -> new DL4JFromDJLTensorWrapperImpl(f));
    }

    @Override
    protected DJLTensor create(DL4JTensor tensor) {
        if(tensor instanceof DL4JFromDJLTensorWrapperImpl) {
            DL4JFromDJLTensorWrapperImpl oth = (DL4JFromDJLTensorWrapperImpl)tensor;
            return oth.getT();
        }
        return new DJLFromDL4JTensorWrapperImpl(tensor);
    }

    @Override
    protected DL4JTensor extract(DJLTensor tensor) {
        if(tensor instanceof DJLFromDL4JTensorWrapperImpl) {
            DJLFromDL4JTensorWrapperImpl oth = (DJLFromDL4JTensorWrapperImpl)tensor;
            return oth.getT();
        }
        return new DL4JFromDJLTensorWrapperImpl(tensor);
    }
    @Override
    protected DJLTensorOperations createData(DL4JTensorOperations data) {
        return new DJLTensorOperationsImpl(data);
    }

    @Override
    protected DL4JTensorOperations extractData(DJLTensorOperations djlTensorOperations) {
        return new DL4JTensorOperationsImpl(djlTensorOperations);
    }

    @Override
    public DJLTensor get() {
        return this;
    }


    @Override
    public PtNDArray getNDArray() {
        return (PtNDArray) DJLTensorFactory.getManager().create(getDataAsFloatArray(), DJLTensorFactory.getShape(size()));
    }

    @Override
    public DJLTensor apply(DifferentiableUnaryOperator<DJLTensor, DJLTensorOperations, Size> differentiableUnaryOperator) {
        DJLTensor tensor = new DJLTensorImpl(this).apply(differentiableUnaryOperator);
        return tensor;
    }

    @Override
    public DJLTensor apply(DifferentiableBinaryOperator<DJLTensor, DJLTensorOperations, Size> differentiableBinaryOperator, DJLTensor ml4JTensor) {
        return new DJLTensorImpl(this).apply(differentiableBinaryOperator, ml4JTensor);
    }

    @Override
    public ValueNode<DJLTensor> getValueNode() {
        return new ValueNodeWrapper<>(t.getValueNode(), f -> new DJLFromDL4JTensorWrapperImpl(f).requires_grad_(false), f -> new DL4JFromDJLTensorWrapperImpl(f).requires_grad_(false));
    }

    @Override
    public GradNode<DJLTensor> getGradNode() {
        return new GradNodeWrapper<>(t.getGradNode(), f -> new DJLFromDL4JTensorWrapperImpl(f).requires_grad_(false), f -> new DL4JFromDJLTensorWrapperImpl(f).requires_grad_(false));
    }


    @Override
    public DL4JTensor toDL4JTensor() {
        return new DL4JTensorImpl(this);
    }

    @Override
    public ML4JTensor toML4JTensor(DirectedComponentsContext context) {
        return null;
    }

    @Override
    public DJLTensor toDJLTensor() {
        return new DJLTensorImpl(this);
    }
}
