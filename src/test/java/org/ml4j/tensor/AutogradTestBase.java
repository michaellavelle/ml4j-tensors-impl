package org.ml4j.tensor;

import org.junit.Assert;
import org.junit.Test;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.BackwardConfig;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;


public abstract class AutogradTestBase<V extends Tensor<V, D>, D extends TensorOperations<D>> extends TestBase<V, D> {

    protected abstract boolean isNativeGradientSupported();

    protected abstract boolean isNativeGradientExpected();


    @Override
    protected abstract D createData(float value);

    @Override
    protected abstract D createData(float value, Size size);

    protected abstract V createRandomValue(boolean requires_grad, int... dims);
    protected abstract V createOnesValue(boolean requires_grad, int... dims);

    @Test
    public void test_scalartensor_addition() {
        var a = createRandomValue(true, 2, 2);

        //var a = torch.randn(2, 2).requires_grad_(true);
        var b = createRandomValue(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertTrue(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));
        assertTrue(b.grad().size().dimensions().length == 0);
        assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createGradValue(1, false, new Size(2, 2)).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }

    }

    @Test
    public void test_scalartensor_addition_second_without_requires_grad() {
        var a = createRandomValue(true, 2, 2);
        var b = createRandomValue(false);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertTrue(a.requires_grad());
        assertFalse(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));

        assertTrue(a.requires_grad());
        assertFalse(b.requires_grad());

        Assert.assertNull(b.grad());

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
        }
    }


    @Test
    public void test_scalartensor_addition_first_without_requires_grad() {
        var a = createRandomValue(false, 2, 2);
        var b = createRandomValue(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertTrue(c.requires_grad());
        assertFalse(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));

        assertTrue(b.grad().size().dimensions().length == 0);
        assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        Assert.assertNull(a.grad());

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }
    }


    @Test
    public void test_scalartensor_addition_reversed() {
        var a = createRandomValue(true, 2, 2);
        var b = createRandomValue(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = b.add(a);

        assertTrue(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));

        assertTrue(b.grad().size().dimensions().length == 0);
        assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }

    }


    //


    @Test
    public void test_both_scalartensor_addition() {
        var a = createRandomValue(true);
        var b = createRandomValue(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertTrue(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false).mul(2f));

        assertTrue(b.grad().size().dimensions().length == 0);
        assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 2f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }
    }

    @Test
    public void test_both_scalartensor_addition_second_without_requires_grad() {
        var a = createRandomValue(true);
        var b = createRandomValue(false);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertTrue(a.requires_grad());
        assertFalse(b.requires_grad());

        c.backward(createOnesValue(false).mul(2f));

        assertTrue(a.requires_grad());
        assertFalse(b.requires_grad());

        Assert.assertNull(b.grad());

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
        }
    }

    protected void assertArrayEqual(float[] actual, float[] expected, float delta) {
        Assert.assertArrayEquals(expected, actual, delta);
    }

    @Test
    public void test_both_scalartensor_addition_first_without_requires_grad() {
        var a = createRandomValue(false);
        var b = createRandomValue(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertTrue(c.requires_grad());
        assertFalse(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false).mul(2f));

        assertTrue(b.grad().size().dimensions().length == 0);
        assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 2f, 0.001f);

        Assert.assertNull(a.grad());

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }
    }


    @Test
    public void test_both_scalartensor_addition_reversed() {
        var a = createRandomValue(true);
        var b = createRandomValue(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = b.add(a);

        assertTrue(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false).mul(2f));

        assertTrue(b.grad().size().dimensions().length == 0);
        assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 2f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }
    }

    @Test
    public void test_scalarbroadcast_addition() {
        var a = createRandomValue(true, 2, 2);
        var b = createRandomValue(true);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }
        
        var c = a.add(b);


        assertTrue(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));

        assertTrue(b.grad().size().dimensions().length == 0);
        assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }
    }



    @Test
    public void test_scalarbroadcast_addition_second_without_requires_grad() {
        var a = createRandomValue(true, 2, 2);
        var b = createRandomValue(false, 1, 1);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }
        var c = a.add(b);

        assertTrue(a.requires_grad());
        assertFalse(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));

        assertTrue(a.requires_grad());
        assertFalse(b.requires_grad());

        Assert.assertNull(b.grad());

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
        }
    }


    @Test
    public void test_scalarbroadcast_addition_first_without_requires_grad() {
        var a = createRandomValue(false, 2, 2);
        var b = createRandomValue(true, 1, 1);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertTrue(c.requires_grad());
        assertFalse(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));

        assertTrue(b.grad().size().dimensions().length == 2);
        assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        Assert.assertNull(a.grad());

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }
    }


    @Test
    public void test_scalarbroadcast_addition_reversed() {
        var a = createRandomValue(true, 2, 2);
        var b = createRandomValue(true, 1, 1);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = b.add(a);
        System.out.println("Result size:" + c.size());

        assertTrue(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));

        assertTrue(b.grad().size().dimensions().length == 2);
        assertTrue(b.grad().numel() == 1);
        Assert.assertEquals(b.grad().getDataAsFloatArray()[0], 8f, 0.001f);

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }
    }


    @Test
    public void test_tensor_addition() {
        var a = createRandomValue(true, 2, 2);
        var b = createRandomValue(true, 2, 2);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertTrue(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));


        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(b.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
        }
    }

    @Test
    public void test_tensor_addition_second_without_requires_grad() {
        var a = createRandomValue(true, 2, 2);
        var b = createRandomValue(false, 2, 2);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertTrue(a.requires_grad());
        assertFalse(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));

        assertTrue(a.requires_grad());
        assertFalse(b.requires_grad());

        Assert.assertNull(b.grad());

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
        }
   
    }

    @Test
    public void test_tensor_addition_first_without_requires_grad() {
        var a = createRandomValue(false, 2, 2);
        var b = createRandomValue(true, 2, 2);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }
        var c = a.add(b);

        assertTrue(c.requires_grad());
        assertFalse(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));

        assertArrayEqual(b.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(b.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        Assert.assertNull(a.grad());

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }
    }

    @Test
    public void test_tensor_addition_reversed() {
        var a = createRandomValue(true, 2, 2);
        var b = createRandomValue(true, 2, 2);

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
            b.getGradNode().setDisableNativeGradient(true);
        }

        var c = b.add(a);

        assertTrue(a.requires_grad());
        assertTrue(b.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));


        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(b.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), b.grad().isNativeGradient());
        }
    }


    @Test
    public void test_scalar_addition() {
        var a = createRandomValue(true, 2, 2);
        var b = (float) Math.random();

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertTrue(a.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));

        assertTrue(a.requires_grad());

        assertArrayEqual(a.grad().getDataAsFloatArray(), createOnesValue(false, 2, 2).mul(2f).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
        }
    }

    @Test(expected = IllegalStateException.class)
    public void test_scalar_addition_without_requires_grad() {
        var a = createRandomValue(false, 2, 2);
        var b = (float) Math.random();

        if (!isNativeGradientExpected()) {
            a.getGradNode().setDisableNativeGradient(true);
        }

        var c = a.add(b);

        assertFalse(a.requires_grad());
        assertFalse(c.requires_grad());

        c.backward(createOnesValue(false, 2, 2).mul(2f));
       
        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
        }
   
    }


    @Test
    public void test_requires_grad_inplace() {
        var a = createRandomValue(false, 5, 5);
        var b = createRandomValue(true, 5, 5);

        a = a.add(b);

        assertTrue(a.requires_grad());

        // non-leaf
        a = createRandomValue(false, 5, 5).add(0f);
        b = createRandomValue(true, 5, 5);
        a = a.add(b);
        assertTrue(a.requires_grad());

    }

    @Test
    public void test_hessian_vector() {

        var x = createRandomValue(true, 2, 2);
        var y = createRandomValue(true, 2, 2);

        if (!isNativeGradientExpected()) {
            x.getGradNode().setDisableNativeGradient(true);
            y.getGradNode().setDisableNativeGradient(true);
        }

        var z = x.mul(x).add(y.mul(x).add(y.mul(y)));
        z.backward(createOnesValue(false, 2, 2), new BackwardConfig().with_keep_graph(true)); // create_graph=True

        //with torch.no_grad():
        x.requires_grad_(false);
        y.requires_grad_(false);

        var x_grad = x.mul(2).add(y);
        var y_grad = x.add(y.mul(2));

        assertArrayEqual(x.grad().getDataAsFloatArray(), x_grad.getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(y.grad().getDataAsFloatArray(), y_grad.getDataAsFloatArray(), 0.0001f);

        x.requires_grad_(true);
        y.requires_grad_(true);

        var grad_sum = x.grad().mul(2).add(y.grad());

        grad_sum.backward(createOnesValue(false, 2, 2));
        var x_hv = createOnesValue(false, 2, 2).mul(5); // Should be ones not zeros with create graph
        var y_hv = createOnesValue(false, 2, 2).mul(4); // Should be ones not zeros with create graph

        assertArrayEqual(x.grad().getDataAsFloatArray(), x_grad.add(x_hv).getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(y.grad().getDataAsFloatArray(), y_grad.add(y_hv).getDataAsFloatArray(), 0.0001f);

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), x.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), y.grad().isNativeGradient());
        }
    }

    @Test(expected = IllegalStateException.class)
    public void test_hessian_vector_without_create_graph() {

        var x = createRandomValue(true, 2, 2);
        var y = createRandomValue(true, 2, 2);

        if (!isNativeGradientExpected()) {
            x.getGradNode().setDisableNativeGradient(true);
            y.getGradNode().setDisableNativeGradient(true);
        }

        var z = x.mul(x).add(y.mul(x).add(y.mul(y)));

        z.backward(createOnesValue(false, 2, 2)); // create_graph=False

        //with torch.no_grad():
        x.requires_grad_(false);
        y.requires_grad_(false);

        var x_grad = x.mul(2).add(y);
        var y_grad = x.add(y.mul(2));

        assertArrayEqual(x.grad().getDataAsFloatArray(), x_grad.getDataAsFloatArray(), 0.0001f);
        assertArrayEqual(y.grad().getDataAsFloatArray(), y_grad.getDataAsFloatArray(), 0.0001f);

        x.requires_grad_(true);
        y.requires_grad_(true);

        var grad_sum = x.grad().mul(2).add(y.grad());

        grad_sum.backward(createOnesValue(false, 2, 2));

        if (isNativeGradientSupported()) {
            Assert.assertEquals(isNativeGradientExpected(), x.grad().isNativeGradient());
            Assert.assertEquals(isNativeGradientExpected(), y.grad().isNativeGradient());
        }
    }
}
