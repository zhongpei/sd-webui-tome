import launch


if not launch.is_installed("tomesd"):
    print('Installing requirements for tomesd')
    launch.run_pip("install tomesd", "requirements for tomesd")
