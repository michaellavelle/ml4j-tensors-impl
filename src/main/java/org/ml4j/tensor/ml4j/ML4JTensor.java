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

package org.ml4j.tensor.ml4j;

import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.Node;
import org.ml4j.nn.components.DirectedComponentsContext;
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
public class ML4JTensor extends AutogradValueImpl<ML4JTensor, ML4JTensorOperations, Size> implements AutogradValue<ML4JTensor, ML4JTensorOperations, Size>, DifferentiableWrappedTensorOperations<ML4JTensor, ML4JTensorOperations>, TensorOperations<ML4JTensor>, org.ml4j.autograd.DataSupplier<ML4JTensorOperations>, Tensor<ML4JTensor, ML4JTensorOperations> {

	private DirectedComponentsContext context;

	public ML4JTensor(DirectedComponentsContext context, Supplier<ML4JTensorOperations> data, Size size) {
		this(context, data, size, new ArrayList<>());
	}

	public ML4JTensor(DirectedComponentsContext context, float data, Size size) {
		this(context, () -> new ML4JTensorOperationsImpl(context, data, size), size, new ArrayList<>());
	}

	protected ML4JTensor(DirectedComponentsContext context, Supplier<ML4JTensorOperations> data, Size size, List<Node<?>> children) {
		super(data, size, children);
		this.context = context;
	}

	@Override
	public int numel() {
		return size().numel();
	}

	@Override
	public ML4JTensor sum() {
		throw new UnsupportedOperationException();
	}

	@Override
	public ML4JTensor mean() {
		throw new UnsupportedOperationException();
	}

	@Override
	public ML4JTensor norm() {
		throw new UnsupportedOperationException();
	}

	@Override
	public ML4JTensor columnSums() {
		throw new UnsupportedOperationException();
	}

	@Override
	public ML4JTensor rowSums() {
		throw new UnsupportedOperationException();
	}

	@Override
	public ML4JTensor cloneTensor() {
		throw new UnsupportedOperationException();
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
	public ML4JTensor size_(Size size) {
		throw new UnsupportedOperationException();
	}

	@Override
	public ML4JTensor zero_() {
		//throw new UnsupportedOperationException();
		return this;
	}

	@Override
	public ML4JTensor normal_(float v1, float v2) {

		//throw new UnsupportedOperationException();
		return this;
	}

	@Override
	public ML4JTensor fill_(float value) {
		return this;
		//throw new UnsupportedOperationException();
	}

	@Override
	public ML4JTensor view(Size size) {
		throw new UnsupportedOperationException();
	}

	@Override
	public ML4JTensor view(int... dims) {
		if (dims.length == 1 && dims[0] == -1) {
			return applyUnaryOperator(t -> t.view(-1), (g, v) -> g.view(size()), "view", s -> new Size(s.numel()));
		}
		throw new UnsupportedOperationException();
	}

	@Override
	public void close() {

	}

	@Override
	protected ML4JTensor createAutogradValue(Supplier<ML4JTensorOperations> data, Size size, List<Node<?>> children) {
		return new ML4JTensor(context, data, size, children);
	}

	@Override
	protected ML4JTensor getInitialInstance() {
		return this;
	}

	@Override
	protected Supplier<ML4JTensorOperations> multiplicativeIdentity() {
		return () -> new ML4JTensorOperationsImpl(context, 1, size());
	}

	@Override
	protected Supplier<ML4JTensorOperations> additiveIdentity() {
		return () -> new ML4JTensorOperationsImpl(context, 0, size());
	}

	@Override
	public ML4JTensor add(ML4JTensor other) {
		return applyBinaryOperator(other, (f, s) -> f.add(s), (g, p) -> g, (g, p) -> g, "add", (f, s) -> f);
	}

	@Override
	public ML4JTensor get() {
		return this;
	}
}
