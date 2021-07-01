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
import org.ml4j.tensor.dl4j.DL4JTensor;
import org.ml4j.tensor.dl4j.DL4JTensorImpl;
import org.ml4j.tensor.ml4j.*;

public class DJLFromML4JTensorWrapperImpl extends TensorWrapperImpl<ML4JTensor, DJLTensor, ML4JTensorOperations, DJLTensorOperations> implements DJLTensor {

    private DirectedComponentsContext context;
    private GradNode<DJLTensor> gradNode;
    private ValueNode<DJLTensor> valueNode;

    @Override
    public void backward() {
        super.backward();
    }

    public DJLFromML4JTensorWrapperImpl(DirectedComponentsContext context, ML4JTensor t) {
        super(t);
        this.context = context;
        this.gradNode =  new GradNodeWrapper<>(t.getGradNode(), f -> new DJLFromML4JTensorWrapperImpl(context, f), f -> new ML4JFromDJLTensorWrapperImpl(context, f));
        this.valueNode =  new ValueNodeWrapper<>(t.getValueNode(), f -> new DJLFromML4JTensorWrapperImpl(context, f), f -> new ML4JFromDJLTensorWrapperImpl(context, f));
    }

    @Override
    protected DJLTensor create(ML4JTensor tensor) {
        if(tensor instanceof ML4JFromDJLTensorWrapperImpl) {
            ML4JFromDJLTensorWrapperImpl oth = (ML4JFromDJLTensorWrapperImpl)tensor;
            return oth.getT();
        }
        return new DJLFromML4JTensorWrapperImpl(context, tensor);
    }

    @Override
    protected ML4JTensor extract(DJLTensor tensor) {
        if(tensor instanceof DJLFromML4JTensorWrapperImpl) {
            DJLFromML4JTensorWrapperImpl oth = (DJLFromML4JTensorWrapperImpl)tensor;
            return oth.getT();
        }
        return new ML4JFromDJLTensorWrapperImpl(context, tensor);
    }
    @Override
    protected DJLTensorOperations createData(ML4JTensorOperations data) {
        return new DJLTensorOperationsImpl(data);
    }

    @Override
    protected ML4JTensorOperations extractData(DJLTensorOperations djlTensorOperations) {
        return new ML4JTensorOperationsImpl(context, djlTensorOperations);
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
        return new ValueNodeWrapper<>(t.getValueNode(), f -> new DJLFromML4JTensorWrapperImpl(context, f).requires_grad_(false), f -> new ML4JFromDJLTensorWrapperImpl(context, f).requires_grad_(false));
    }

    @Override
    public GradNode<DJLTensor> getGradNode() {
        return new GradNodeWrapper<>(t.getGradNode(), f -> new DJLFromML4JTensorWrapperImpl(context, f).requires_grad_(false), f -> new ML4JFromDJLTensorWrapperImpl(context, f).requires_grad_(false));
    }

    @Override
    public DL4JTensor toDL4JTensor() {
        return new DL4JTensorImpl(this);
    }

    @Override
    public ML4JTensor toML4JTensor(DirectedComponentsContext context) {
        return new ML4JTensorImpl(this, context);
    }

    @Override
    public DJLTensor toDJLTensor() {
        return new DJLTensorImpl(this);
    }
}
