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

package org.ml4j.tensor.dl4j;

import ai.djl.ndarray.types.Shape;
import org.jvmpy.symbolictensors.MultiplicationRules;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.AutogradValueRegistry;
import org.ml4j.autograd.impl.AutogradValueProperties;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.tensor.DifferentiableWrappedTensorOperations;
import org.ml4j.tensor.Tensor;
import org.ml4j.tensor.TensorOperations;
import org.ml4j.tensor.djl.DJLFromDL4JTensorWrapperImpl;
import org.ml4j.tensor.djl.DJLTensor;
import org.ml4j.tensor.djl.DJLTensorImpl;
import org.ml4j.tensor.ml4j.ML4JTensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Optional;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * @author Michael Lavelle
 */
public class DL4JTensorImpl extends DifferentiableWrappedTensorOperations<DL4JTensor, DL4JTensorOperations> implements AutogradValue<DL4JTensor, DL4JTensorOperations, Size>, TensorOperations<DL4JTensor>, org.ml4j.autograd.DataSupplier<DL4JTensorOperations>, Tensor<DL4JTensor, DL4JTensorOperations>, DL4JTensor {

	public DL4JTensorImpl(INDArray indArray, boolean requires_grad, AutogradValueRegistry registry) {
		super(() -> new DL4JTensorOperationsImpl(indArray), new AutogradValueProperties<Size>().setRegistry(registry).setContext(getSize(indArray.shape())).setRequires_grad(requires_grad));
	}

	private static Size getSize(long[] dimsArray) {
		int[] dims = new int[dimsArray.length];
		for (int i = 0; i < dims.length; i++) {
			dims[i] = (int)dimsArray[i];
		}
		return new Size(dims);
	}

	public <X extends AutogradValue<X, Y, Z>, Y, Z> DL4JTensorImpl(AutogradValue<X, Y, Z> other, Function<Y, DL4JTensorOperations> dataMapper, Function<Z, Size> contextMapper, Function<X, DL4JTensor> valueMapper, Function<DL4JTensor, X> valueReverseMapper, Supplier<Optional<DL4JTensor>> nativeGradientSupplier, String name) {
		super(other, dataMapper, contextMapper, valueMapper, valueReverseMapper, nativeGradientSupplier);
		name_(name);
	}

	public DL4JTensorImpl(DL4JTensor other) {
		super(other);
	}

	public DL4JTensorImpl(DJLTensor other) {
		this(other, da -> da == null ? null : new DL4JTensorOperationsImpl(da), s -> s, d -> d == null ? null : new DL4JTensorImpl(d), m -> m == null ? null : new DJLTensorImpl(m), null, other.name());
	}

	public DL4JTensorImpl(float data, AutogradValueProperties<Size> properties) {
		this(() -> new DL4JTensorOperationsImpl(createArray(data, properties.getContext(), properties.isRequires_grad())), properties);
	}

	public INDArray getNDArray() {
		return data().get().getNDArray();
	}

	/*
	//@Override
	public DL4JTensor size_(Size size) {
		this.context = size;
		return self();
	}
	 */


	@Override
	public DL4JTensor getTensor(int[]... ranges) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensor matmul(DL4JTensor other) {
		Size origSize = this.size();
		Size[] sizes = MultiplicationRules.matmul(size(), other.size());
		return this.applyBinaryOperator(other, (f, s) -> f.matmul(s), (g, p) -> {
			Size origGSize = sizes[3];
			DL4JTensor r = g.reshape(sizes[2]).matmul(p.getRight().t());
			//g.resize_(size());
			return r;
		}, (g, p) -> {
			Size origGSize = sizes[3];
			Size origLeftSize = origSize;
			DL4JTensor r = g.reshape(sizes[2]).t().matmul(p.getLeft().reshape(sizes[0])).t();
			//g.reshape_(origGSize);
			//p.getLeft().reshape_(origLeftSize);
			return r;
		}, "matmul", (f, s) -> {
			Size result =  sizes[3];
			int[] dims = result.dimensions();
			int [] firstDims = new int[dims.length- 1];
			for (int i = 0; i < firstDims.length; i++) {
				firstDims[i] = dims[i];
			}
			return sizes[3];
		});
	}



	private static INDArray createArray(float data, Size size, boolean requires_grad) {
		INDArray arr = Nd4j.ones(size.dimensions()).mul(data);
		return arr;
	}

	public static Shape getShape(Size size) {
		long[] s = new long[size.getDimensions().size()];
		for (int i = 0; i < s.length; i++) {
			s[i] = size.getDimensions().get(i);
		}
		return new Shape(s);
	}

	public DL4JTensorImpl(Supplier<DL4JTensorOperations> data, AutogradValueProperties<Size> properties) {
		super(data, properties);
		requires_grad_(properties.isRequires_grad());
	}

	@Override
	protected DL4JTensor getSub(DL4JTensor other, Size size, float scale) {
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

			INDArray matrixOld = other.data().get().getNDArray();
			Size s = scalar ? new Size() : new Size(newDims);
			INDArray matrix = s.dimensions().length == 0 ? Nd4j.scalar(data[0]) : Nd4j.create(data, s.dimensions());
			DL4JTensorOperations ops = new DL4JTensorOperationsImpl(matrix);
			return new DL4JTensorImpl(() -> ops, new AutogradValueProperties<Size>().setContext(s).setRequires_grad(requires_grad()).setRegistry(properties().getRegistry()).setCreate_graph(create_graph()));
		}
	}

	@Override
	protected void close(DL4JTensorOperations dl4JTensorOperations) {

	}

	@Override
	protected DL4JTensor createAutogradValue(Supplier<DL4JTensorOperations> data, AutogradValueProperties<Size> properties) {
		return new DL4JTensorImpl(data, properties);
	}

	@Override
	protected DL4JTensor getInitialInstance() {
		return this;
	}

	@Override
	protected Supplier<DL4JTensorOperations> multiplicativeIdentity() {
		return () -> new DL4JTensorOperationsImpl(size(), 1);
	}

	@Override
	protected Supplier<DL4JTensorOperations> additiveIdentity() {
		return () -> new DL4JTensorOperationsImpl(size(), 0);
	}

	@Override
	public DL4JTensor get() {
		return this;
	}

	@Override
	public DL4JTensor toDL4JTensor() {
		return this;
	}

	@Override
	public ML4JTensor toML4JTensor(DirectedComponentsContext context) {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public DJLTensor toDJLTensor() {
		return new DJLTensorImpl(new DJLFromDL4JTensorWrapperImpl(this));
	}
}
