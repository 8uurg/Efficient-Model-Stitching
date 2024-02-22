#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <grpc/grpc.h>
// - Server
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
// - Client
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/support/async_unary_call.h>
#include <grpcpp/support/status.h>

#include "ealib.grpc.pb.h"
#include "ealib.pb.h"
#pragma GCC diagnostic pop