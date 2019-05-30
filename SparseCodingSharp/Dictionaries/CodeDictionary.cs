using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Text;
using Numpy;

namespace SparseCodingSharp.Dictionaries
{
    /// <summary>
    /// The CodeDictionary class is more or less a wrapper around the NumSharp array class. It holds a NumSharp ndarray in
    /// the attribute `matrix` and adds some useful functions for it.The dictionary elements can be accessed
    /// either by D.matrix[i, j] or directly through D[i, j].
    /// </summary>
    public class CodeDictionary
    {
        protected NDarray _matrix;
        protected int[] _shape;

        public CodeDictionary(NDarray matrix)
        {
            _matrix = np.array(matrix);
            _shape = matrix.shape.Dimensions;
            if (_shape.Length != 2)
                throw new ArgumentException("given matrix shape must be two dimensional");
        }

        public NDarray this[params int[] select]
        {
            get { return _matrix[select]; }
        }

        /// <summary>
        /// Checks whether the dictionary is unitary.
        ///<returns>True, if the dicitonary is unitary.</returns>
        /// </summary>
        public bool IsUnitary
        {
            get
            {
                var n = _shape[0];
                var K = _shape[1];
                if (n == K)
                    return np.allclose(np.dot(_matrix.transpose(), _matrix), np.eye(n));
                else
                    return false;
            }
        }

        /// <summary>
        /// Checks wheter the dictionary is l2-normalized.
        /// <returns>True, if dictionary is l2-normalized.</returns>
        /// </summary>
        public bool IsNormalized
        {
            get
            {
                var n = _shape[0];
                var K = _shape[1];
                var norms = new double[K];
                for (int i = 0; i < K; i++)
                    norms[i] = np.linalg.norm(_matrix[":", i]);
                return np.allclose(norms, np.ones(K));

            }
        }

        public static double[] MutualCoherence(NDarray D)
        {
            var (n, K) = (D.shape.Dimensions[0], D.shape.Dimensions[1]);
            var mu = new List<double>();
            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < K; j++)
                {
                    if (j == i) continue;
                        mu.Add( Math.Abs((double)np.dot(D[":", i].T, D[":", j]) / (np.linalg.norm(D[":", i]) * np.linalg.norm(D[":", j]))));
                }
            }
            return mu.ToArray();
        }

        /// <summary>
        /// Transforms the dictionary columns into patches and orders them for plotting purposes.
        /// </summary>
        /// <returns>Reordered dictionary matrix</returns>
        NDarray to_img()
        {
            // dictionary dimensions
            var D = this._matrix;
            var (n, K) = (D.shape.Dimensions[0], D.shape.Dimensions[1]);
            var M = this._matrix;
            //# stretch atoms
            //for k in range(K):
            //    M[:, k] = M[:, k] - (M[:, k].min())
            //    if M[:, k].max():
            //        M[:, k] = M[:, k] / D[:, k].max()

            // patch size
            var n_r = (int) (Math.Sqrt(n));

            // patches per row / column
            var K_r = (int) (Math.Sqrt(K));

            //# we need n_r*K_r+K_r+1 pixels in each direction
            var dim = n_r * K_r + K_r + 1;
            var V = np.ones(dim, dim) * (float)np.amin(D);

            //# compute the patches
            //patches = [np.reshape(D[:, i], (n_r, n_r)) for i in range(K)]

            //# place patches
            //for i in range(K_r):
            //    for j in range(K_r):
            //        V[j * n_r + 1 + j:(j + 1) * n_r + 1 + j, i * n_r + 1 + i:(i + 1) * n_r + 1 + i] = patches[
            //            i * K_r + j]
            return V;
        }
    }
}
