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

package org.ml4j.tensor;

import org.apache.commons.lang3.tuple.Pair;
import org.jvmpy.symbolictensors.MultiplicationRules;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.Value;
import org.ml4j.autograd.arithmetic.operations.ArithmeticOperations;
import org.ml4j.autograd.arithmetic.operations.DifferentiableWrappedArithmeticOperations;
import org.ml4j.tensor.djl.DJLTensor;
import org.ml4j.tensor.ml4j.ML4JTensor;

import java.util.function.BiFunction;
import java.util.function.BinaryOperator;

public interface DifferentiableWrappedTensorOperations<V extends TensorOperations<V> & Value<V, D, Size>, D extends TensorOperations<D>> extends DifferentiableWrappedArithmeticOperations<V, D, Size>, AutogradValue<V, D, Size>, TensorOperations<V> {

	@Override
	default V relu() {
		return applyUnaryOperator(D::relu, (g, v) -> g.mul(v.gt(0)), "gt", s -> s);
	}

	@Override
	default V view(Size size) {
		return applyUnaryOperator(v -> v.view(size), (g, v) -> g.view(size()), "view", s -> size);
	}

	@Override
	default V norm() {
		return applyUnaryOperator(t -> t.norm(), (g, v) -> backwardNotYetImplemented(), "norm", s -> new Size());
	}

	@Override
	default V sum() {
		return applyUnaryOperator(t -> t.sum(), (g, v) -> v.mul(g), "sum", s -> new Size());
	}

	@Override
	default V mean() {
		return applyUnaryOperator(t -> t.mean(), (g, v) -> { return v.mul(0).add(1).mul(g).div(v.numel()); }, "mean", s -> new Size());
	}

	default V backwardNotYetImplemented() {
		throw new UnsupportedOperationException();
	}

	default V t() {
		return applyUnaryOperator(D::t, (g, v) -> g.t(), "t", s -> s.t());
	}

	@Override
	default V sigmoid() {
		return applyUnaryOperator(D::sigmoid, (g, v) -> g.mul(sigGrad(v.getDataAsFloatArray()[0])), "gt", s -> s);
	}

	@Override
	default V reshape_(Size size) {
		return applyUnaryOperator(f -> f.reshape_(size), (g, v) -> g.reshape_(size()), "reshape", s -> size);
	}

	@Override
	default V mul_(V other) {
		return applyInlineBinaryOperator(other, D::mul_, "mul");
	}

	@Override
	default V matmul(V other) {

		Size[] sizes = MultiplicationRules.matmul(size(), other.size());

		return this.applyBinaryOperator(other, (f, s) -> f.reshape_(sizes[0]).matmul(s.reshape_(sizes[1])), (g, p) -> {
			return g.reshape_(sizes[2]).matmul(p.getRight().reshape_(sizes[1]).t()).reshape_(size());
		}, (g, p) -> {
			 return g.reshape_(sizes[2]).t().matmul(p.getLeft().reshape_(sizes[0])).t().reshape_(other.size());
		}, "matmul", (f, s) -> {
			Size result =  sizes[3];
			int[] dims = result.dimensions();
			int [] firstDims = new int[dims.length- 1];
			for (int i = 0; i < firstDims.length; i++) {
				firstDims[i] = dims[i];
			}
			return new Size(new Size(firstDims), new Size(dims[dims.length - 1]));
		});
	}

	@Override
	default Size getMappedContext(Size f, Size s) {
		Size size = MultiplicationRules.getBroadcast(f, s);
		return size;
	}

	@Override
	default V bernoulli() {
		return applyUnaryOperator(D::bernoulli, (g, v) -> g, "gt", s -> s);
	}

	@Override
	default int numel() {
		return size().numel();
	}


	@Override
	default V columnSums() {
		throw new UnsupportedOperationException();
	}


	@Override
	default V normal_(float v1, float v2) {
		// Sun
		return self();
		//throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	default V fill_(float value) {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	default int size(int dim) {
		return size().getDimensions().get(dim);
	}
	@Override
	default Size size() {
		return context();
	}

	@Override
	default V zero_() {
		// Sun
		return self();
		//throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	default V view(int... dims) {
		if (dims.length == 1 && dims[0] == -1) {
			return applyUnaryOperator(t -> t.view(-1), (g, v) -> g.view(size()), "view", s -> new Size(s.numel()));
		}
		throw new UnsupportedOperationException();
	}

	@Override
	default V rowSums() {
		throw new UnsupportedOperationException();
	}

	private float sig(float x) {
		return 1f / (1f + (float)Math.exp(-x));
	}

	private float sigGrad(float x) {
		float s = sig(x);
		return s * ( 1 - s);
	}

	@Override
	default float[] getDataAsFloatArray() {
		return this.data().get().getDataAsFloatArray();
	}
}
