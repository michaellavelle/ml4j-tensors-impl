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

import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.jni.JniUtils;
import org.jvmpy.symbolictensors.MultiplicationRules;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.AutogradValueRegistry;
import org.ml4j.autograd.BackwardConfig;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.impl.AutogradValueProperties;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.Node;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.tensor.DifferentiableWrappedTensorOperations;
import org.ml4j.tensor.Tensor;
import org.ml4j.tensor.TensorOperations;
import org.ml4j.tensor.dl4j.DL4JFromDJLTensorWrapperImpl;
import org.ml4j.tensor.dl4j.DL4JTensor;
import org.ml4j.tensor.dl4j.DL4JTensorImpl;
import org.ml4j.tensor.ml4j.ML4JFromDJLTensorWrapperImpl;
import org.ml4j.tensor.ml4j.ML4JTensor;
import org.ml4j.tensor.ml4j.ML4JTensorImpl;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * @author Michael Lavelle
 */
public class DJLTensorImpl extends DifferentiableWrappedTensorOperations<DJLTensor, DJLTensorOperations> implements AutogradValue<DJLTensor, DJLTensorOperations, Size>, TensorOperations<DJLTensor>, org.ml4j.autograd.DataSupplier<DJLTensorOperations>, Tensor<DJLTensor, DJLTensorOperations>, DJLTensor {


	public DJLTensorImpl(NDArray ndArray, boolean requires_grad, AutogradValueRegistry registry) {
		super(() -> new DJLTensorOperationsImpl(ndArray), new AutogradValueProperties<Size>().setContext(getSize(ndArray.getShape())).setRegistry(registry).setRequires_grad(requires_grad));
	}

	private static Size getSize(Shape shape) {
		int[] dims = new int[shape.getShape().length];
		for (int i = 0; i < dims.length; i++) {
			dims[i] = (int)shape.getShape()[i];
		}
		return new Size(dims);
	}

	public <X extends AutogradValue<X, Y, Z>, Y, Z> DJLTensorImpl(AutogradValue<X, Y, Z> other, Function<Y, DJLTensorOperations> dataMapper, Function<Z, Size> contextMapper, Function<X, DJLTensor> valueMapper, Function<DJLTensor, X> valueReverseMapper, Supplier<Optional<DJLTensor>> nativeGradientSupplier, String name) {
		super(other, dataMapper, contextMapper, valueMapper, valueReverseMapper, nativeGradientSupplier);
		name_(name);
	}

	public DJLTensorImpl(DJLTensor other) {
		super(other);
	}

	public DJLTensorImpl(ML4JTensor other) {
		this(other, da -> da == null ? null : new DJLTensorOperationsImpl(da), s -> s, d -> d == null ? null : new DJLTensorImpl(d), m -> m == null ? null : new ML4JTensorImpl(m, other.getDirectedComponentsContext()), null, null);
	}

	public DJLTensorImpl(DL4JTensor other) {
		this(other, da -> da == null ? null : new DJLTensorOperationsImpl(da), s -> s, d -> d == null ? null : new DJLTensorImpl(d), m -> m == null ? null : new DL4JTensorImpl(m), null, other.name());
	}

	public DJLTensorImpl(float data, AutogradValueProperties<Size> properties) {
		super(() -> new DJLTensorOperationsImpl(createArray(data, properties.getContext(), properties.isRequires_grad())), properties);
	}

	public PtNDArray getNDArray() {
		return (PtNDArray)data().get().getNDArray();
	}

	@Override
	public DL4JTensor toDL4JTensor() {
		return new DL4JFromDJLTensorWrapperImpl(this);
	}

	@Override
	public ML4JTensor toML4JTensor(DirectedComponentsContext context) {
		return new ML4JFromDJLTensorWrapperImpl(context, this);
	}

	@Override
	public DJLTensor toDJLTensor() {
		return this;
	}

	/*
	@Override
	public DJLTensor size_(Size size) {
		this.context = size;
		return self();
	}

	 */


	@Override
	public void backward(DJLTensor g, BackwardConfig config) {
		try {
			JniUtils.backward(getNDArray(), g.getNDArray(), config.keep_graph(), config.keep_graph());
		} catch (EngineException e) {
			throw new IllegalStateException(e);
		}

		super.backward(g, config);
	}

	@Override
	public DJLTensor matmul(DJLTensor other) {

		Size origSize = this.size();
		Size[] sizes = MultiplicationRules.matmul(size(), other.size());
		return this.applyBinaryOperator(other, (f, s) -> f.matmul(s), (g, p) -> {
			Size origGSize = sizes[3];
			DJLTensor r = g.reshape(sizes[2]).matmul(p.getRight().t());
			//resize_(origGSize);
			return r.resize_(size());
		}, (g, p) -> {
			Size origGSize = sizes[3];
			Size origLeftSize = origSize;
			DJLTensor r = g.reshape(sizes[2]).t().matmul(p.getLeft().reshape(sizes[0])).t().reshape(other.size());
			//g.resize_(origGSize);
			//p.getLeft().resize_(origLeftSize);
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

	@Override
	public DJLTensor requires_grad_(boolean requires_grad) {
		if (requires_grad) {
			data().get().getNDArray().setRequiresGradient(true);
		}
		super.requires_grad_(requires_grad);
		getGradNode().setNativeGradientSupplier(createNativeGradient());
		return this;
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
			return () -> {NDArray grad = data().get().getNDArray().getGradient(); return grad.sum().getFloat() == 0 ? Optional.empty() : Optional.of(new DJLTensorImpl(() -> {DJLTensorOperationsImpl data = new DJLTensorOperationsImpl(grad); data.setNativeGradient(true); return data;}, new AutogradValueProperties<Size>().setContext(size()).setRegistry(properties().getRegistry()).setName("nativeGrad"))); };
		} else {
			return () -> Optional.empty();
		}
	}

	@Override
	public String toString() {
		return name();
	}

	public static Shape getShape(Size size) {
		long[] s = new long[size.getDimensions().size()];
		for (int i = 0; i < s.length; i++) {
			s[i] = size.getDimensions().get(i);
		}
		return new Shape(s);
	}

	public DJLTensorImpl(Supplier<DJLTensorOperations> data, AutogradValueProperties<Size> properties) {
		super(data, properties);
		requires_grad_(properties.isRequires_grad());
		getGradNode().setNativeGradientSupplier(createNativeGradient());
		name_(properties.getName());
	}

	@Override
	protected DJLTensor getSub(DJLTensor other, Size size, float scale) {
		if (scale == 1) {
			return other;
		} else {

			if (other.size().dimensions()[0] == 2 && other.size().dimensions()[1] == 128 && other.size().dimensions()[2] == 65
			&& size.dimensions()[0] == 1 && size.dimensions()[1] == 65) {
				return other.getTensor(new int[] {0, 1}, new int[] {0,1}, new int[]{ 0, 65}).resize_(new Size(1, 65));
			}

			boolean scalar = size.dimensions().length == 0;
			int div = scale == 2 ? 2 : (int) Math.sqrt(scale);
			int[] dims = other.size().dimensions();
			int prod = 1;
			int[] newDims = new int[dims.length];
			int[][] ranges = new int[newDims.length][2];

			for (int i = 0; i < newDims.length; i++) {
				newDims[i] = dims[i] /div;
				prod = prod * newDims[i];
				if (scale == 2) {
					div = 1;
				}
				ranges[i][0] = 0;
				ranges[i][1] = newDims[i];
			}

			/*
			float[] oldData = other.getDataAsFloatArray();
			System.out.println("PROD:" + prod + ":" + oldData.length);
			float[] data = new float[prod];
			int ind = 0;
			int newInd = 0;
			for (int i = 0; i < dims.length; i++) {
				for (int j = 0; j < dims[i]; j++) {
					if (j < newDims[i]) {
						if (newInd < data.length && ind < oldData.length) {
							data[newInd] = oldData[ind];
							System.out.println("Setting:" + newInd + ":" + ind);
						}
						newInd++;
					}
					ind++;
				}
			}

			 */

			Size s = scalar ? new Size() : new Size(newDims);
		if (size.dimensions().length != s.dimensions().length) {
			throw new IllegalStateException();
		} else {
			for (int i = 0; i < size.dimensions().length; i++) {
				if (s.dimensions()[i] != size.dimensions()[i]) {
					throw new IllegalStateException();
				}
			}
		}

			if (scalar) {
				return other.getTensor(newDims);
			} else {
				return other.getTensor(ranges);
			}
			//NDArray matrix = DJLTensorFactory.getManager().create(data, getShape(s));
			//DJLTensorOperations ops = new DJLTensorOperationsImpl(matrix);
			//return new DJLTensorImpl(() -> ops, s, requires_grad(), create_graph);
		}
	}

	@Override
	protected void close(DJLTensorOperations djlTensorOperations) {
		djlTensorOperations.close();
	}

	@Override
	protected DJLTensor createAutogradValue(Supplier<DJLTensorOperations> data, AutogradValueProperties<Size> properties) {
		return new DJLTensorImpl(data, properties);
	}

	@Override
	protected DJLTensor getInitialInstance() {
		return this;
	}

	@Override
	protected Supplier<DJLTensorOperations> multiplicativeIdentity() {
		return () -> new DJLTensorOperationsImpl(getShape(size()), 1, false);
	}

	@Override
	protected Supplier<DJLTensorOperations> additiveIdentity() {
		return () -> new DJLTensorOperationsImpl(getShape(size()), 0, false);
	}

	@Override
	public DJLTensor get() {
		return this;
	}
}
