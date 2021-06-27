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

import org.junit.Assert;
import org.junit.Test;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.tensor.TensorTestBase;
import org.ml4j.tensor.djl.DJLFromML4JTensorWrapperImpl;
import org.ml4j.tensor.djl.DJLTensor;

public class ML4JTensorTest extends TensorTestBase<ML4JTensor, ML4JTensorOperations> {

	private static MatrixFactory matrixFactory = new JBlasRowMajorMatrixFactory();

	private static DirectedComponentsContext context =  new DirectedComponentsContextImpl(matrixFactory, true);

	@Test
	public void switchTest() {

		var a = createGradValue(-4f, true, new Size(2, 2)).name_("a");

		var b = createGradValue(-4f, true, new Size(2, 2)).name_("a");

		if (!isNativeGradientExpected()) {
			a.getGradNode().setDisableNativeGradient(true);
		}

		var c = a.add(b);

		ML4JTensorImpl t = (ML4JTensorImpl)c;

		DJLTensor s = new DJLFromML4JTensorWrapperImpl(ML4JTensorFactory.DEFAULT_DIRECTED_COMPONENTS_CONTEXT, t);

		var u = s.mul(s);

		assertEquals(createData(-8f, new Size(2, 2)), c.data().get());

		u.backward();

		assertEquals(createData(-16f, new Size(2, 2)), t.grad().data().get());

		assertEquals(createData(-16f, new Size(2, 2)), a.grad().data().get());

		if (isNativeGradientSupported()) {
			Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
		}
	}


	@Override
	protected ML4JTensorImpl createGradValue(float value, boolean requires_grad) {
        return new ML4JTensorImpl(context, () -> createData(value), size, requires_grad, false);
	}

	@Override
	protected ML4JTensorImpl createGradValue(float value, boolean requires_grad, Size size) {
		return new ML4JTensorImpl(context, () -> createData(value, size), size, requires_grad, false);
	}

	@Override
	protected ML4JTensorImpl createGradValue(ML4JTensorOperations value, boolean requires_grad) {
        return new ML4JTensorImpl(context, () -> value, size, requires_grad, false);
	}

	@Override
	protected ML4JTensorOperations createData(float value) {
		return new ML4JTensorOperationsImpl(context, value, size);
	}

	@Override
	protected ML4JTensorOperations createData(float value, Size size) {
		return new ML4JTensorOperationsImpl(context, value, size);
	}

	@Override
	protected void assertEquals(ML4JTensorOperations value1, ML4JTensorOperations value2) {
		Matrix m1 = value1.getMatrix();
		Matrix m2 = value2.getMatrix();
		Assert.assertEquals(m1.getLength(), m2.getLength());
		for (int i = 0; i < m1.getLength(); i++) {
			Assert.assertEquals(m1.get(i), m2.get(i), 0.01f);
		}
	}

	@Override
	protected ML4JTensorOperations add(ML4JTensorOperations value1, ML4JTensorOperations value2) {
		return value1.add(value2);
	}

	@Override
	protected ML4JTensorOperations mul(ML4JTensorOperations value1, float value2) {
		return value1.mul(value2);
	}


	@Override
	protected void assertSize(ML4JTensor tensor, Size s) {
		Assert.assertEquals(tensor.size().dimensions().length, s.dimensions().length);
		for (int i = 0; i < s.dimensions().length; i++) {
			Assert.assertEquals(tensor.size().dimensions()[i], s.dimensions()[i]);

		}
		/*
		if (s.dimensions().length == 2) {
			Assert.assertEquals(tensor.data().get().getMatrix().getRows(), s.dimensions()[0]);
			Assert.assertEquals(tensor.data().get().getMatrix().getColumns(), s.dimensions()[1]);
		} else if (s.dimensions().length == 1) {
			Assert.assertEquals(tensor.data().get().getMatrix().getRows(), 1);
			Assert.assertEquals(tensor.data().get().getMatrix().getColumns(), s.dimensions()[0]);
		} else {
			throw new IllegalArgumentException();
		}
		*/
	}
	@Override
	protected boolean isNativeGradientSupported() {
		return false;
	}

	@Override
	protected boolean isNativeGradientExpected() {
		return false;
	}
}
