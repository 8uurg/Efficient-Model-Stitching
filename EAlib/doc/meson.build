# Based on https://github.com/wayland-project/weston/blob/main/doc/sphinx/meson.build
# Adjusted to potentially work on Windows as well :)

sphinx = find_program('sphinx-build', required: true)
doxygen = find_program('doxygen', required: true)
breathe = find_program('breathe-apidoc', required: true)

output_doxygen = meson.current_build_dir() + '/doxygen'

sphinx_conf_data = configuration_data()
sphinx_conf_data.set('BUILD_ROOT', output_doxygen)
sphinx_conf_data.set('VERSION', meson.project_version())
sphinx_conf = configure_file(
		input: 'conf.py.in',
		output: 'conf.py',
		configuration: sphinx_conf_data
)

doxy_conf_data = configuration_data()
doxy_conf_data.set('SRC_ROOT', meson.source_root())
doxy_conf_data.set('OUTPUT_DIR', output_doxygen)
doxygen_conf_weston = configure_file(
		input: 'doxygen.ini.in',
		output: 'doxygen.ini',
		configuration: doxy_conf_data
)

script_data = configuration_data()
script_data.set('SRCDIR', meson.source_root() + '/doc')
script_data.set('CONFDIR', meson.current_build_dir())
script_data.set('OUTDIR', meson.current_build_dir() + '/doc')

script_data.set('DOXYGEN_CONF', meson.current_build_dir() + '/doxygen.ini')
script_data.set('DOXYGEN_CMD', doxygen.path())
script_data.set('SPHINX_CMD', sphinx.path())

if (host_machine.system() == 'windows')
    script_doxy_sphinx = configure_file(
        input: 'run_doxygen_sphinx.bat.in',
        output: 'run_doxygen_sphinx.bat',
        configuration: script_data
    )
else
    script_doxy_sphinx = configure_file(
        input: 'run_doxygen_sphinx.sh.in',
        output: 'run_doxygen_sphinx.sh',
        configuration: script_data
    )
endif

sphinx_doc = custom_target(
		'ealib-doc-breathe',
		command: script_doxy_sphinx,
		output: 'doc',
		build_by_default: true,
)

docs = run_target(
		'docs',
		command: script_doxy_sphinx,
)
