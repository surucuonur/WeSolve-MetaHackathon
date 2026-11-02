// [harness] Written by Meta Muse Spark (muse-spark-1.3-contributor) via the Meta Model API.
// [harness] Call muse-spark/calls/0002-kernels-routing-combine-a1, response resp_6ae913db4181fa96e7db464c.
// [harness] --- Muse Spark output below, unmodified ---
uint idx = thread_position_in_grid.x;
uint r = idx / 768;
uint c = idx % 768;
float w0 = weights[r * 4 + 0];
float w1 = weights[r * 4 + 1];
float w2 = weights[r * 4 + 2];
float w3 = weights[r * 4 + 3];
const device bfloat* yrow = y + r * 3072;
float y0 = static_cast<float>(yrow[0 * 768 + c]);
float y1 = static_cast<float>(yrow[1 * 768 + c]);
float y2 = static_cast<float>(yrow[2 * 768 + c]);
float y3 = static_cast<float>(yrow[3 * 768 + c]);
float sum = w0 * y0 + w1 * y1 + w2 * y2 + w3 * y3;
bfloat sum_bf = static_cast<bfloat>(sum);
float sh = static_cast<float>(shared[r * 768 + c]);
float res = static_cast<float>(sum_bf) + sh;
out[idx] = static_cast<bfloat>(res);
