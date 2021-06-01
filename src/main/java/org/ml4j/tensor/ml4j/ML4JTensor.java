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
	public ML4JTensor applyBinaryOperator(ML4JTensor other, BinaryOperator<ML4JTensorOperations> forward, BiFunction<ML4JTensor, Pair<ML4JTensor, ML4JTensor>, ML4JTensor> backThis, BiFunction<ML4JTensor, Pair<ML4JTensor, ML4JTensor>, ML4JTensor> backOther, String op, BinaryOperator<Size> contextMapper) {
		if (!size().getDimensions().equals(other.size().getDimensions())) {
			try {
				Size broadcastSize = MultiplicationRules.getBroadcast(size(), other.size());
				System.out.println("Broadcast:" + broadcastSize);
				System.out.println("BroadcastA:" + size() + ":" + other.size());

				if (broadcastSize.getDimensions().equals(size().getDimensions()) || broadcastSize.getDimensions().equals(other.size().getDimensions())) {
					float scale1 = (float) broadcastSize.numel() / (float) size().numel();
					float scale2 = (float) broadcastSize.numel() / (float) other.size().numel();
					System.out.println("Scale:" + scale1 + ":" + scale2);
					return super.applyBinaryOperator(other, forward, (g, p) -> getSub(backThis.apply(g, p), size(), scale1).mul(scale1), (g, p) -> getSub(backOther.apply(g, p), other.size(), scale2).mul(scale2), op, (f, s) -> broadcastSize);
				} else {
					throw new RuntimeException();
				}
			} catch (IllegalArgumentException e) {
				// Size cannot be broadcast, perhaps it's matMul.
			}
		}
		return super.applyBinaryOperator(other, forward, backThis, backOther, op, contextMapper);
	}

	private ML4JTensor getSub(ML4JTensor other, Size size, float scale) {
		if (scale == 1) {
			return other;
		} else {
			boolean scalar = size.dimensions().length == 0;
			int div = (int) Math.sqrt(scale);
			int[] dims = other.size().dimensions();
			int prod = 1;
			System.out.println("Size:" + size + ":" + other.size());
			int[] newDims = new int[dims.length];
			for (int i = 0; i < newDims.length; i++) {
				System.out.println("div:" + dims[i] + ":" + div);
				newDims[i] = dims[i] /div;
				System.out.println("ND:" + newDims[i]);
				prod = prod * newDims[i];
			}
			float[] oldData = other.getDataAsFloatArray();
			float[] data = new float[prod];
			int ind = 0;
			int newInd = 0;
			System.out.println("Prod:" + prod + ":" + dims.length + ":" + newDims.length);
			for (int i = 0; i < dims.length; i++) {
				for (int j = 0; j < dims[i]; j++) {
					if (j < newDims[i]) {
						if (newInd < data.length && ind < oldData.length) {
							System.out.println(newInd + ":" + ind);
							data[newInd] = oldData[ind];
						}
						newInd++;

					}
					ind++;
				}
			}

			Matrix matrixOld = other.data().get().getMatrix();
			System.out.println("Matrix old:" + matrixOld.getRows() + ":" + matrixOld.getColumns());
			System.out.println("Data length:" + data.length);
			Matrix matrix = context.getMatrixFactory().createMatrixFromRowsByRowsArray(matrixOld.getRows() / (int)div, matrixOld.getColumns() / div, data);
			Size s = scalar ? new Size() : new Size(newDims);
			ML4JTensorOperations ops = new ML4JTensorOperationsImpl(context, matrix, s);
			System.out.println("Result:" + s);
			return new ML4JTensor(context, () -> ops, s);
		}
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

	/*
	@Override
	public ML4JTensor add(ML4JTensor other) {
		return applyBinaryOperator(other, (f, s) -> f.add(s), (g, p) -> g, (g, p) -> g, "add", fgetMappedContext(f, s));
	}
	 */

	@Override
	public ML4JTensor get() {
		return this;
	}
}
