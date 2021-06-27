package org.ml4j.tensor;

import org.jvmpy.symbolictensors.Size;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;

public class TensorMatrixFactory<T extends Tensor<T, D>, D> implements MatrixFactory {

    private TensorFactory<T, D> tensorFactory;

    public TensorMatrixFactory(TensorFactory<T, D> tensorFactory) {
        this.tensorFactory = tensorFactory;
    }

    protected TensorMatrix create(T tensor) {
        return new TensorMatrix(tensor);
    }

    @Override
    public Matrix createOnes(int rows, int columns) {
        return create(tensorFactory.ones(new Size(rows, columns)));
    }

    @Override
    public Matrix createOnes(int length) {
        return create(tensorFactory.ones(new Size(length)));
    }

    @Override
    public Matrix createZeros(int rows, int columns) {
        return create(tensorFactory.zeros(new Size(rows, columns)));
    }

    @Override
    public Matrix createRandn(int rows, int columns) {
        return create(tensorFactory.randn(new Size(rows, columns)));
    }

    @Override
    public Matrix createRand(int rows, int columns) {
        return create(tensorFactory.rand(new Size(rows, columns)));
    }

    @Override
    public Matrix createMatrixFromRows(float[][] floats) {
        float[] data = new float[floats.length * floats[0].length];
        int index = 0;
        for (int r = 0; r < floats.length; r++) {
            for (int c = 0; c < floats[r].length; c++) {
                data[index++] = floats[r][c];
            }
        }
        return createMatrixFromRowsByRowsArray(floats.length, floats[0].length, data);
    }

    @Override
    public Matrix createMatrix(int rows, int columns) {
        return create(tensorFactory.empty(new Size(rows, columns)));
    }

    @Override
    public Matrix createMatrixFromColumnsByColumnsArray(int i, int i1, float[] floats) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Matrix createMatrixFromRowsByRowsArray(int rows, int columns, float[] floats) {
        System.out.println("Creating:" + rows + ":" + columns + ":" + floats.length);
        return create(tensorFactory.create(floats, new Size(rows, columns)));
    }

    @Override
    public Matrix createMatrix() {
        return create(tensorFactory.create());
    }

    @Override
    public Matrix createMatrix(float[] data) {
        return create(tensorFactory.create(data));
    }

    @Override
    public Matrix createHorizontalConcatenation(Matrix matrix, Matrix matrix1) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Matrix createVerticalConcatenation(Matrix matrix, Matrix matrix1) {
        throw new UnsupportedOperationException();
    }
}
