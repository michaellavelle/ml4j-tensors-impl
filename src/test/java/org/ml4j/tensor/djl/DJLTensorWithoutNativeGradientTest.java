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
import org.junit.Assert;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.impl.AutogradValueProperties;
import org.ml4j.tensor.TensorTestBase;

public class DJLTensorWithoutNativeGradientTest extends TensorTestBase<DJLTensor, DJLTensorOperations> {

	@Override
	protected DJLTensorImpl createGradValue(float value, boolean requires_grad) {
        return new DJLTensorImpl(() -> createData(value), new AutogradValueProperties<Size>().setRegistry(registry).setContext(size).setRequires_grad(requires_grad));
	}

	private Shape getShape(int...dims) {
		long[] d = new long[dims.length];
		for (int i = 0; i < d.length; i++){
			d[i] = dims[i];
		}
		return new Shape(d);
	}

	@Override
	protected DJLTensor createGradValue(float[] data, int... dims) {
		return new DJLTensorImpl(DJLTensorFactory.getManager().create(data, getShape(dims)), false, registry);
	}

	@Override
	protected DJLTensor createGradValue(DJLTensorOperations value, boolean requires_grad) {
        return new DJLTensorImpl(() -> value, new AutogradValueProperties<Size>().setContext(size).setRegistry(registry).setRequires_grad(requires_grad)).requires_grad_(requires_grad);
	}

	@Override
	protected DJLTensor createGradValue(float value, boolean requires_grad, Size size) {
		return new DJLTensorImpl(() -> createData(value, size), new AutogradValueProperties<Size>().setContext(size).setRegistry(registry).setRequires_grad(requires_grad));
	}

	@Override
	protected DJLTensorOperations createData(float value) {
		return new DJLTensorOperationsImpl(DJLTensorImpl.getShape(size), value, false);
	}

	@Override
	protected DJLTensorOperations createData(float value, Size size) {
		return new DJLTensorOperationsImpl(DJLTensorImpl.getShape(size), value, false);
	}

	@Override
	protected void assertEquals(DJLTensorOperations value1, DJLTensorOperations value2) {
		float[] m1 = value1.getNDArray().toFloatArray();
		float[] m2 = value2.getNDArray().toFloatArray();
		Assert.assertEquals(m1.length,  m2.length);
		for (int i = 0; i < m1.length; i++) {

			Assert.assertEquals(m1[i], m2[i], 0.01f);
		}
	}

	@Override
	protected void assertSize(DJLTensor tensor, Size s) {
		Assert.assertEquals(tensor.size().dimensions().length, s.dimensions().length);
		for (int i = 0; i < s.dimensions().length; i++) {
			Assert.assertEquals(tensor.size().dimensions()[i], s.dimensions()[i]);

		}
		Assert.assertEquals(tensor.getNDArray().getShape().getShape().length, s.dimensions().length);
		for (int i = 0; i < s.dimensions().length; i++) {
			Assert.assertEquals(tensor.getNDArray().getShape().getShape()[i], s.dimensions()[i]);

		}
	}

	@Override
	protected DJLTensorOperations add(DJLTensorOperations value1, DJLTensorOperations value2) {
		return value1.add(value2);
	}

	@Override
	protected DJLTensorOperations mul(DJLTensorOperations value1, float value2) {
		return value1.mul(value2);
	}

	@Override
	protected boolean isNativeGradientSupported() {
		return true;
	}

	@Override
	protected boolean isNativeGradientExpected() {
		return false;
	}
}
