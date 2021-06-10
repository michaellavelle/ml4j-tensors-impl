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

import org.apache.commons.lang3.tuple.Pair;
import org.jvmpy.symbolictensors.MultiplicationRules;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.Matrix;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.Node;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.tensor.DifferentiableWrappedTensorOperations;
import org.ml4j.tensor.Tensor;
import org.ml4j.tensor.TensorBase;
import org.ml4j.tensor.TensorOperations;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;

/**
 * An AutogradValue implementation that supports the operations defined by TensorOperations,
 * and that takes advantage of the fact that the wrapped data also implements TensorOperations
 * by implementing default DifferentiableWrappedTensorOperations methods.
 *
 * @author Michael Lavelle
 */
public class ML4JTensor extends TensorBase<ML4JTensor, ML4JTensorOperations> implements AutogradValue<ML4JTensor, ML4JTensorOperations, Size>, DifferentiableWrappedTensorOperations<ML4JTensor, ML4JTensorOperations>, TensorOperations<ML4JTensor>, org.ml4j.autograd.DataSupplier<ML4JTensorOperations>, Tensor<ML4JTensor, ML4JTensorOperations> {

	private DirectedComponentsContext context;

	public ML4JTensor(DirectedComponentsContext context, Supplier<ML4JTensorOperations> data, Size size, boolean requires_grad, boolean create_graph) {
		this(context, data, size, new ArrayList<>(), requires_grad, create_graph);
	}

	public ML4JTensor(DirectedComponentsContext context, float data, Size size, boolean requires_grad, boolean create_graph) {
		this(context, () -> new ML4JTensorOperationsImpl(context, data, size), size, new ArrayList<>(), requires_grad, create_graph);
	}

	protected ML4JTensor(DirectedComponentsContext context, Supplier<ML4JTensorOperations> data, Size size, List<Node<?>> children, boolean requires_grad, boolean create_graph) {
		super(data, size, children, requires_grad, create_graph);
		this.context = context;
	}

	@Override
	protected ML4JTensor getSub(ML4JTensor other, Size size, float scale) {
		if (scale == 1) {
			return other;
		} else {
			boolean scalar = size.dimensions().length == 0;
			int div = (int) Math.sqrt(scale);
			int[] dims = other.size().dimensions();
			int prod = 1;
			int[] newDims = new int[dims.length];
			for (int i = 0; i < newDims.length; i++) {
				newDims[i] = dims[i] /div;
				prod = prod * newDims[i];
			}
			float[] oldData = other.getDataAsFloatArray();
			float[] data = new float[prod];
			int ind = 0;
			int newInd = 0;
			for (int i = 0; i < dims.length; i++) {
				for (int j = 0; j < dims[i]; j++) {
					if (j < newDims[i]) {
						if (newInd < data.length && ind < oldData.length) {
							data[newInd] = oldData[ind];
						}
						newInd++;

					}
					ind++;
				}
			}

			Matrix matrixOld = other.data().get().getMatrix();
			Matrix matrix = context.getMatrixFactory().createMatrixFromRowsByRowsArray(matrixOld.getRows() / (int)div, matrixOld.getColumns() / div, data);
			Size s = scalar ? new Size() : new Size(newDims);
			ML4JTensorOperations ops = new ML4JTensorOperationsImpl(context, matrix, s);
			return new ML4JTensor(context, () -> ops, s, requires_grad(), create_graph);
		}
	}

	@Override
	public ML4JTensor cloneTensor() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int size(int dim) {
		return size().getDimensions().get(dim);
	}

	@Override
	public ML4JTensor size_(Size size) {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public void close() {
		// No-op for now.
	}

	@Override
	protected ML4JTensor createAutogradValue(Supplier<ML4JTensorOperations> data, Size size, List<Node<?>> children, boolean requires_grad, boolean create_graph) {
		return new ML4JTensor(context, data, size, children, requires_grad, create_graph);
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
	public ML4JTensor get() {
		return this;
	}
}
