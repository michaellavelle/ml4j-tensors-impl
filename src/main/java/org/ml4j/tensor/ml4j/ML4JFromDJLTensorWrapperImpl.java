package org.ml4j.tensor.ml4j;

import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.impl.GradNodeWrapper;
import org.ml4j.autograd.impl.ValueNodeWrapper;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.DifferentiableBinaryOperator;
import org.ml4j.autograd.operators.DifferentiableUnaryOperator;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.tensor.TensorWrapperImpl;
import org.ml4j.tensor.djl.*;

public class ML4JFromDJLTensorWrapperImpl extends TensorWrapperImpl<DJLTensor, ML4JTensor, DJLTensorOperations, ML4JTensorOperations> implements ML4JTensor {

    private DirectedComponentsContext context;

    public ML4JFromDJLTensorWrapperImpl(DirectedComponentsContext context, DJLTensor t) {
        super(t);
        this.context = context;
    }

    @Override
    public DirectedComponentsContext getDirectedComponentsContext() {
        return context;
    }

    @Override
    protected DJLTensor extract(ML4JTensor tensor) {
        if(tensor instanceof ML4JFromDJLTensorWrapperImpl) {
            ML4JFromDJLTensorWrapperImpl oth = (ML4JFromDJLTensorWrapperImpl)tensor;
            return oth.t;
        }
        return new DJLFromML4JTensorWrapperImpl(context, tensor);
    }

    @Override
    protected ML4JTensor create(DJLTensor tensor) {
        if(tensor instanceof DJLFromML4JTensorWrapperImpl) {
            DJLFromML4JTensorWrapperImpl oth = (DJLFromML4JTensorWrapperImpl)tensor;
            return oth.getT();
        }
        return new ML4JFromDJLTensorWrapperImpl(context, tensor);
    }

    @Override
    protected ML4JTensorOperations createData(DJLTensorOperations data) {
        return new ML4JTensorOperationsImpl(context, data);
    }

    @Override
    protected DJLTensorOperations extractData(ML4JTensorOperations ml4JTensorOperations) {
        return new DJLTensorOperationsImpl(ml4JTensorOperations);
    }

    @Override
    public ML4JTensor get() {
        return this;
    }

    @Override
    public ValueNode<ML4JTensor> getValueNode() {
        return new ValueNodeWrapper<>(t.getValueNode(), f -> new ML4JFromDJLTensorWrapperImpl(context, f).requires_grad_(false), f -> new DJLFromML4JTensorWrapperImpl(context, f).requires_grad_(false));
    }

    @Override
    public GradNode<ML4JTensor> getGradNode() {
        return new GradNodeWrapper<>(t.getGradNode(), f -> f == null ? null : new ML4JFromDJLTensorWrapperImpl(context, f).requires_grad_(false), f -> f == null ? null : new DJLFromML4JTensorWrapperImpl(context, f).requires_grad_(false));
    }

    @Override
    public ML4JTensor apply(DifferentiableUnaryOperator<ML4JTensor, ML4JTensorOperations, Size> differentiableUnaryOperator) {
        ML4JTensor tensor = new ML4JTensorImpl(this, context).apply(differentiableUnaryOperator);
        return tensor;
    }

    @Override
    public ML4JTensor apply(DifferentiableBinaryOperator<ML4JTensor, ML4JTensorOperations, Size> differentiableBinaryOperator, ML4JTensor ml4JTensor) {
        return new ML4JTensorImpl(this, context).apply(differentiableBinaryOperator, ml4JTensor);
    }
}
