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

import ai.djl.ndarray.types.Shape;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.Node;
import org.ml4j.tensor.DifferentiableWrappedTensorOperations;
import org.ml4j.tensor.Tensor;
import org.ml4j.tensor.TensorOperations;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;


/**
 * An AutogradValue implementation that supports the operations defined by TensorOperations,
 * and that takes advantage of the fact that the wrapped data also implements TensorOperations
 * by implementing default DifferentiableWrappedTensorOperations methods.
 *
 * @author Michael Lavelle
 */
public class DJLTensor extends AutogradValueImpl<DJLTensor, DJLTensorOperations, Size> implements AutogradValue<DJLTensor, DJLTensorOperations, Size>, DifferentiableWrappedTensorOperations<DJLTensor, DJLTensorOperations>, TensorOperations<DJLTensor>, org.ml4j.autograd.DataSupplier<DJLTensorOperations>, Tensor<DJLTensor, DJLTensorOperations> {

	public DJLTensor(Supplier<DJLTensorOperations> data, Size size) {
		this(data, size, new ArrayList<>());
	}

	public DJLTensor(float data, Size size) {
		this(() -> new DJLTensorOperationsImpl(DJLTensorFactory.getManager().ones(getShape(size)).mul(data)), size, new ArrayList<>());
	}

	public static Shape getShape(Size size) {
		long[] s = new long[size.getDimensions().size()];
		for (int i = 0; i < s.length; i++) {
			s[i] = size.getDimensions().get(i);
		}
		return new Shape(s);
	}

	protected DJLTensor(Supplier<DJLTensorOperations> data, Size size, List<Node<?>> children) {
		super(data, size, children);
	}


	@Override
	public int numel() {
		return size().numel();
	}

	@Override
	public DJLTensor sum() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor mean() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor norm() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor columnSums() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor rowSums() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor cloneTensor() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor matmul(DJLTensor other) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor t() {
		return applyUnaryOperator((f, s) -> f.t(), 0, (g, p) -> g, "t", f -> f.t());
	}

	@Override
	public Size size() {
		return context();
	}

	@Override
	public int size(int dim) {
		return size().getDimensions().get(dim);
	}
	@Override
	public DJLTensor size_(Size size) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor zero_() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor normal_(float v1, float v2) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor fill_(float value) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor view(Size size) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DJLTensor view(int... dims) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void close() {

	}

	@Override
	protected DJLTensor createAutogradValue(Supplier<DJLTensorOperations> data, Size size, List<Node<?>> children) {
		return new DJLTensor(data, size, children);
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
