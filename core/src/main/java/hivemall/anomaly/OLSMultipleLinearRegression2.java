package hivemall.anomaly;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

public final class OLSMultipleLinearRegression2 extends OLSMultipleLinearRegression {

    public OLSMultipleLinearRegression2() {
        super();
    }

    public OLSMultipleLinearRegression2(double threshold) {
        super(threshold);
    }

    @Override
    protected void validateSampleData(double[][] x, double[] y)
            throws MathIllegalArgumentException {
        if ((x == null) || (y == null)) {
            throw new NullArgumentException();
        }
        if (x.length != y.length) {
            throw new DimensionMismatchException(y.length, x.length);
        }
        if (x.length == 0) {  // Must be no y data either
            throw new NoDataException();
        }
        if (x[0].length > x.length) {
            throw new MathIllegalArgumentException(
                    LocalizedFormats.NOT_ENOUGH_DATA_FOR_NUMBER_OF_PREDICTORS,
                    x.length, x[0].length);
        }
    }

}
