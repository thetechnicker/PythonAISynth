#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

int main(int argc, char *argv[])
{
    tensorflow::Session *session;
    tensorflow::Status status;
    status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok())
    {
        std::cout << status.ToString() << "\n";
        return 1;
    }

    // Read in the protobuf graph we exported
    // (The path seems to be relative to the cwd. Keep this in mind
    // when calling `bazel run`.)
    tensorflow::GraphDef graph_def;
    status = ReadBinaryProto(tensorflow::Env::Default(), "../../tmp/test/saved_model.pb", &graph_def);
    if (!status.ok())
    {
        std::cout << status.ToString() << "\n";
        return 1;
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok())
    {
        std::cout << status.ToString() << "\n";
        return 1;
    }
}
