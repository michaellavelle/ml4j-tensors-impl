/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.tensor.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.jni.JniUtils;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.Node;
import org.ml4j.tensor.DifferentiableWrappedTensorOperations;
import org.ml4j.tensor.Tensor;
import org.ml4j.tensor.TensorBase;
import org.ml4j.tensor.TensorOperations;
import org.ml4j.tensor.ml4j.ML4JTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;


/**
 * @author Michael Lavelle
 */
public class DJLTensor extends TensorBase<DJLTensor, DJLTensorOperations> implements AutogradValue<DJLTensor, DJLTensorOperations, Size>, DifferentiableWrappedTensorOperations<DJLTensor, DJLTensorOperations>, TensorOperations<DJLTensor>, org.ml4j.autograd.DataSupplier<DJLTensorOperations>, Tensor<DJLTensor, DJLTensorOperations> {

	public DJLTensor(Supplier<DJLTensorOperations> data, Size size, boolean requires_grad, boolean create_graph) {
		this(data, size, new ArrayList<>(), requires_grad, create_graph);
	}

	public DJLTensor(float data, Size size, boolean requires_grad, boolean create_graph) {
		this(() -> new DJLTensorOperationsImpl(createArray(data, size, requires_grad)), size, new ArrayList<>(), requires_grad, create_graph);
	}

	private PtNDArray getNDArray() {
		return (PtNDArray)data().get().getNDArray();
	}

	@Override
	public DJLTensor size_(Size size) {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public void backward(DJLTensor g, BackwardConfig config) {
		JniUtils.backward(getNDArray(), g.getNDArray(), false, create_graph);
		super.backward(g, config);
	}

	@Override
	public DJLTensor requires_grad_(boolean requires_grad) {
		if (requires_grad) {
			data().get().getNDArray().setRequiresGradient(true);
		}
		return super.requires_grad_(requires_grad);
	}

	private static NDArray createArray(float data, Size size, boolean requires_grad) {
		NDArray arr = DJLTensorFactory.getManager().ones(getShape(size)).mul(data);
		if (requires_grad) {
			arr.setRequiresGradient(true);
		}
		return arr;
	}

	protected Supplier<Optional<DJLTensor>> createNativeGradient() {
		if (this.requires_grad()) {
			return () -> Optional.of(new DJLTensor(() -> new DJLTensorOperationsImpl(data().get().getNDArray().getGradient()), size(), new ArrayList<>(), false, create_graph));
		} else {
			return () -> Optional.empty();
		}
	}

	public static Shape getShape(Size size) {
		long[] s = new long[size.getDimensions().size()];
		for (int i = 0; i < s.length; i++) {
			s[i] = size.getDimensions().get(i);
		}
		return new Shape(s);
	}

	protected DJLTensor(Supplier<DJLTensorOperations> data, Size size, List<Node<?>> children, boolean requires_grad, boolean create_graph) {
		super(data, size, children, requires_grad, create_graph);
		if (requires_grad) {
			data.get().getNDArray().setRequiresGradient(true);
		}
		getGradNode().setNativeGradientSupplier(createNativeGradient());
	}

	@Override
	protected DJLTensor getSub(DJLTensor other, Size size, float scale) {
		return other;
	}

	@Override
	public DJLTensor cloneTensor() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor view(Size size) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void close() {
		// No-op for now.
	}

	@Override
	protected DJLTensor createAutogradValue(Supplier<DJLTensorOperations> data, Size size, List<Node<?>> children, boolean requires_grad, boolean create_graph) {
		return new DJLTensor(data, size, children, requires_grad, create_graph);
	}

	@Override
	protected DJLTensor getInitialInstance() {
		return this;
	}

	@Override
	protected Supplier<DJLTensorOperations> multiplicativeIdentity() {
		return () -> new DJLTensorOperationsImpl(getShape(size()), 1);
	}

	@Override
	protected Supplier<DJLTensorOperations> additiveIdentity() {
		return () -> new DJLTensorOperationsImpl(getShape(size()), 0);
	}

	@Override
	public DJLTensor get() {
		return this;
	}
}
