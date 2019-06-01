using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Text;
using Numpy;
using Numpy.Models;

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

        public CodeDictionary() { }

        public CodeDictionary(NDarray matrix)
        {
            Initialize(matrix);
        }

        protected void Initialize(NDarray matrix)
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
                    mu.Add(Math.Abs((double)np.dot(D[":", i].T, D[":", j]) / (np.linalg.norm(D[":", i]) * np.linalg.norm(D[":", j]))));
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
            // stretch atoms
            for (int k = 0; k < K; k++)
            {
                M[":", k] = M[":", k] - (M[":", k].min());
                if ((double)M[":", k].max() > 0)
                    M[":", k] = M[":", k] / D[":", k].max();
            }

            // patch size
            var n_r = (int)(Math.Sqrt(n));

            // patches per row / column
            var K_r = (int)(Math.Sqrt(K));

            // we need n_r*K_r+K_r+1 pixels in each direction
            var dim = n_r * K_r + K_r + 1;
            var V = np.ones(dim, dim) * (double)np.amin(D);

            // compute the patches
            var patches = new List<NDarray>();
            for (int i = 0; i < K; i++)
            {
                np.reshape(D[":", i], new Shape(n_r, n_r));
            }

            // place patches
            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < K; j++)
                {
                    V[$"{j * n_r + 1 + j}:{(j + 1) * n_r + 1 + j}, {i * n_r + 1 + i}:{(i + 1) * n_r + 1 + i}"] = patches[i * K_r + j];
                }
            }
            return V;
        }

        /// <summary>
        /// Builds a Dictionary matrix from a given transform
        /// </summary>
        /// <param name="transform">A valid transform(e.g.Haar, DCT-II)</param>
        /// <param name="n">number of rows transform dictionary</param>
        /// <param name="K">number of columns transform dictionary</param>
        /// <param name="normalized">If True, the columns will be l2-normalized</param>
        /// <param name="inverse">Uses the inverse transform(as usually needed in applications)</param>
        /// <returns>Dictionary build from the Kronecker-Delta of the transform applied to the identity.</returns>
        public static NDarray dictionary_from_transform(Func<NDarray, int?, bool, NDarray> transform, int n, int K,
            bool normalized = true, bool inverse = true)
        {
            var H = np.zeros(K, n);
            for (int i = 0; i < n; i++)
            {
                var v = np.zeros(n);
                v[i] = (NDarray)1.0;
                H[":", i] = transform(v, K, normalized);
            }
            if (inverse)
                H = H.T;
            return np.kron(H.T, H.T);
        }
    }

    /// <summary>
    /// A Dictionary based on the IDCT-II transform
    /// </summary>
    public class DCTDictionary : CodeDictionary
    {
        public DCTDictionary(int n, int K)
        {
            NDarray D;
            if (n == K)
                D = unitary_idctii_dictionary(n);
            else if (n < K)
                D = overcomplete_idctii_dictionary(n, K);
            else
                throw new ArgumentException("K has to be as least as large as n.");
            Initialize(D);
        }

        /// <summary>
        /// Build a unitary inverse DCT - II dictionary matrix with K = n
        /// </summary>
        /// <param name="n">square of signal dimension</param>
        /// <returns>Unitary DCT-II dictionary</returns>
        private NDarray unitary_idctii_dictionary(int n)
        {
            return dictionary_from_transform(dctii, n, n, inverse: false);
        }

        /// <summary>
        /// Build an overcomplete inverse DCT - II dictionary matrix with K > n
        /// </summary>
        /// <param name="n">square of signal dimension</param>
        /// <param name="K">square of desired number of atoms</param>
        /// <returns>Overcomplete DCT-II dictionary</returns>
        private NDarray overcomplete_idctii_dictionary(int n, int K)
        {
            if (K > n)
                return dictionary_from_transform(dctii, n, K, inverse: false);
            else
                throw new ArgumentException("K needs to be larger than n.");
        }

        /// <summary>
        /// Computes the inverse discrete cosine transform of type II,
        /// https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
        /// </summary>
        /// <param name="v">Input vector to transform</param>
        /// <param name="sampling_factor">Can be used to "oversample" the input to create overcomplete dictionaries</param>
        /// <param name="normalized">Normalizes the output to make output orthogonal</param>
        /// <returns>Discrete cosine transformed vector</returns>
        public static NDarray dctii(NDarray v, int? sampling_factor = null, bool normalized = true)
        {
            var n = v.shape[0];
            int K = sampling_factor ?? n;
            var array = new double[K];
            for (int k = 0; k < K; k++)
            {
                array[k] = (double)np.sum(np.multiply(v, np.cos(0.5 + (np.arange(n)) * k * np.pi / K)));
            }
            NDarray y = np.array(array);
            if (normalized)
            {
                y[0] = (1.0 / Math.Sqrt(2) * y[0]);
                y = Math.Sqrt(2.0 / n) * y;
            }
            return y;
        }
    }


    /// <summary>
    /// A Dictionary based on the inverse Haar transform
    /// </summary>
    public class HaarDictionary : CodeDictionary
    {
        public HaarDictionary(int n, int K)
        {
            NDarray D;
            if (n == K)
                D = unitary_haar_dictionary(n);
            else if (n < K)
                D = overcomplete_haar_dictionary(n, K);
            else
                throw new ArgumentException("K has to be as least as large as n.");
            Initialize(D);
        }

        /// <summary>
        /// Compute the Haar transform.The code was modified from
        /// https://gist.github.com/tristanwietsma/5667982
        /// </summary>
        /// <param name="v">Input vector to transform</param>
        /// <param name="sampling_factor"> Can be used to "oversample" the input to create overcomplete dictionaries</param>
        /// <param name="inverse">this parameter has no effect</param>
        /// <returns>Haar transformed vector</returns>
        public static NDarray haar(NDarray v, int? sampling_factor = null, bool inverse = false)
        {
            var n = v.shape[0];
            var K = sampling_factor ?? n;
            var tmp = np.zeros(K);
            var count = 2;
            while (count <= n)
            {
                for (int i = 0; i < count / 2; i++)
                {
                    tmp[2 * i] = (v[i] + v[i + (int)(count / 2)]) / Math.Sqrt(2);
                    tmp[2 * i + 1] = (v[i] - v[i + (int)(count / 2)]) / Math.Sqrt(2);
                }
                for (int i = 0; i < count; i++)
                    v[i] = tmp[i];
                count *= 2;
            }
            return np.array(tmp).astype(np.float32);
        }

        /// <summary>
        /// Build a unitary inverse Haar dictionary matrix with K = n
        /// </summary>
        /// <param name="n">square of signal dimension</param>
        /// <returns>Unitary Haar dictionary</returns>
        private NDarray unitary_haar_dictionary(int n)
        {
            return dictionary_from_transform(haar, n, n, inverse: false);
        }

        /// <summary>
        /// Build an overcomplete inverse Haar dictionary matrix with K &gt; n
        /// </summary>
        /// <param name="n">square of signal dimension</param>
        /// <param name="K">square of desired number of atoms</param>
        /// <returns>Overcomplete Haar dictionary</returns>
        private NDarray overcomplete_haar_dictionary(int n, int K)
        {
            if (K > n)
                return dictionary_from_transform(haar, n, K, inverse: false);
            else
                throw new ArgumentException("K needs to be larger than n.");
        }
    }

    public class RandomDictionary : CodeDictionary
    {
        public RandomDictionary(int n, int K)
        {
            NDarray D;
            if (n == K)
                D = random_dictionary(n, n);
            else if (n < K)
                D = random_dictionary(n, K);
            else
                throw new ArgumentException("K has to be as least as large as n.");
            Initialize(D);
        }

        /// <summary>
        /// Build a random dictionary matrix with K = n
        /// </summary>
        /// <param name="n">square of signal dimension</param>
        /// <param name="K">square of desired dictionary atoms</param>
        /// <param name="normalized">If true, columns will be l2-normalized</param>
        /// <param name="seed">Random seed</param>
        /// <returns>Random dictionary</returns>
        private NDarray random_dictionary(int n, int K, bool normalized = true, int? seed = null)
        {
            if (seed.HasValue)
                np.random.seed(seed);
            var H = np.random.rand(n, K) * 255;
            if (normalized)
            {
                for (int k = 0; k < K; k++)
                    H[":", k].imul(1 / np.linalg.norm(H[":", k]));
            }
            return np.kron(H, H);
        }

    }

}
