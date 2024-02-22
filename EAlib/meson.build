project('EAlib', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++17'])

add_global_arguments('-fvisibility=hidden', language: 'cpp')

if get_option('buildtype') in ['debug', 'debugoptimized']
  message('enabling debug symbols')
  add_project_arguments('-g3',  language : 'cpp')
else
  message('not enabling additional debug symbols buildtype is ', get_option('buildtype'))
endif

ssmod = import('sourceset')
pythonmod = import('python')

python = pythonmod.find_installation('python3')

# Dependencies
dep_catch2 = dependency('catch2')
dep_trompeloeil = dependency('trompeloeil', fallback: ['trompeloeil', 'trompeloeil_dep'])
dep_python_embed = python.dependency(embed: true)
dep_python = python.dependency(embed: false)
dep_pybind11 = dependency('pybind11')
dep_cereal = dependency('cereal')

dep_proto = dependency('protobuf')
dep_grpc = dependency('grpc')
dep_grpcpp = dependency('grpc++')
protoc_compiler = find_program('protoc')
grpc_proto_plugin = find_program('grpc_cpp_plugin')
grpc_proto_plugin_path = grpc_proto_plugin.full_path()

# RPC
sources_proto = files([
  'rpc/ealib.proto',
])
generated_proto = [
  'ealib.pb.cc',
  'ealib.pb.h'
]
generated_grpc = [
  'ealib.grpc.pb.cc',
  'ealib.grpc.pb.h'
]

# Compile stuff
sources_protoc_cpp = custom_target(
  'proto_compile',
  input: sources_proto,
  output: generated_proto,
  command : [protoc_compiler, '-I@SOURCE_ROOT@/rpc', '--cpp_out=@OUTDIR@', '@INPUT@'],
)
# 
sources_grpc_cpp = custom_target(
  'grpc_compile',
  input: sources_proto,
  output: generated_grpc,
  command : [protoc_compiler, '-I@SOURCE_ROOT@/rpc', '--grpc_out=@OUTDIR@', f'--plugin=protoc-gen-grpc=@grpc_proto_plugin_path@', '@INPUT@'],
)

dep_lib_rpc = declare_dependency(
  sources: [sources_protoc_cpp, sources_grpc_cpp],
  dependencies : [dep_proto, dep_grpc, dep_grpcpp],
)

stacktrace = false
cpptrace = false
backwardcpp = false

default_config = configuration_data({
  'test': false,
  'pythonembed': true,
  'pythonlib': false,
  'python': true,
})

# General source files
ss = ssmod.source_set()
include_dir = include_directories('include')

# Serialization & Deserialization.
ss.add(dep_cereal)

src_files = files([
  'src/base.cpp',
  'src/logging.cpp',
  'src/acceptation_criteria.cpp',
  'src/initializers.cpp',
  'src/ga.cpp',
  'src/problems.cpp',
  'src/gomea.cpp',
  'src/running.cpp',
  'src/archive.cpp',
  'src/sim.cpp',
  'src/sim-gomea.cpp',
  'src/decentralized.cpp',
  'src/d-gomea.cpp',
  'src/d-ga.cpp',
  'src/ecga.cpp',
  'src/sim-ga.cpp',
  'src/kernels.cpp',
  'src/debughelpers.cpp',
  'src/scalarize.cpp',
])
ss.add(src_files)

# Add RPC requirements
ss.add(dep_lib_rpc)

# Add stacktrace requirements to the build, the configuration (when not commented out)
# uses backtrace, which is a GCC compiler feature, and hence does not work everywhere.
# Set the variable above to true, and uncomment the required includes to allow boosts'
# stacktrace functionality to work.
if (stacktrace)
  cxx = meson.get_compiler('cpp')
  libdl_dep = cxx.find_library('dl')
  backtrace_dep = cxx.find_library('backtrace')
  boost_dep = dependency('boost')
  traceback_deps = [libdl_dep, backtrace_dep, boost_dep]
  
  ss.add(traceback_deps)
endif

if (cpptrace)
  cpptrace_dep = dependency('cpptrace')
  ss.add([cpptrace_dep])
endif

if (backwardcpp)
  backwardcpp_dep = dependency('backward-cpp')
  libdwarf_dep = dependency('libdwarf')
  ss.add([backwardcpp_dep])
endif

# Python sources & dependencies
python_wrapper_files = files([
  'src/python.cpp',
  'src/python/acceptation_criteria.cpp',
  'src/python/archive.cpp',
  'src/python/base.cpp',
  'src/python/ga.cpp',
  'src/python/gomea.cpp',
  'src/python/initializers.cpp',
  'src/python/problems.cpp',
  'src/python/logging.cpp',
  'src/python/running.cpp',
  'src/python/utilities.cpp',
  'src/python/sim.cpp',
  'src/python/sim-gomea.cpp',
  'src/python/sim-ga.cpp',
  'src/python/ecga.cpp',
  'src/python/decentralized.cpp',
  'src/python/d_gomea.cpp',
  'src/python/d_ga.cpp',
])

# Python files
py_files = files([
  'ealib/__init__.py',
  'ealib/decentralized_utilities.py',
])

ss.add(when: 'pythonlib', if_true: python_wrapper_files)
ss.add(when: 'pythonlib', if_true: dep_python)
ss.add(when: 'pythonembed', if_true: dep_python_embed)
ss.add(when: 'python', if_true: dep_pybind11)
# ss.add(dep_capnp)

# Test sources & dependencies
test_case_files = files([
  'test/main.cpp',
  'test/test_base.cpp',
  'test/test_acceptation_criteria.cpp',
  'test/test_initializers.cpp',
  'test/test_utilities.cpp',
  'test/test_ga.cpp',
  'test/test_problems.cpp',
  'test/test_gomea.cpp',
  'test/test_running.cpp',
  'test/test_archive.cpp',
  'test/test_sim.cpp',
  'test/test_sim-gomea.cpp',
  'test/test_ecga.cpp',
  'test/test_sim_ga.cpp',
  'test/test_decentralized.cpp',
  'test/test_d_gomea.cpp',
  'test/test_d_ga.cpp',
  
  'test/test_dispatcher.cpp',
])
ss.add(when: 'test', if_true: test_case_files)
ss.add(when: 'test', if_true: [dep_catch2, dep_trompeloeil])

# Python targets
python_config = configuration_data()
python_config.merge_from(default_config)
python_config.set('pythonlib', true)
python_config.set('pythonembed', false)

ss_python = ss.apply(python_config)
python.extension_module(
  '_ealib',
  ss_python.sources(),
  include_directories: include_dir,
  dependencies: ss_python.dependencies(),
  install: true,
  subdir: 'ealib',
)

python.install_sources(py_files, subdir: 'ealib', pure: false)

# Testing targets
testing_config = configuration_data()
testing_config.merge_from(default_config)
testing_config.set('test', true)

ss_test = ss.apply(testing_config)
test('EAlib tests', executable(
  'test_ealib',
  ss_test.sources(),
  include_directories: include_dir,
  dependencies: ss_test.dependencies(),
  install: false
))


if get_option('build_doc')
  subdir('doc')
endif