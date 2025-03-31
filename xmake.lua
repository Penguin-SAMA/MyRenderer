add_rules("plugin.compile_commands.autoupdate")
set_toolchains("gcc-14")
target("myRenderer")
set_kind("binary")
add_files("src/*.cpp")
add_includedirs("include", { public = true })
if is_plat("macosx") then
	add_includedirs("/opt/homebrew/include", "/opt/homebrew/include/SDL2")
	add_linkdirs("/opt/homebrew/lib")
end
set_languages("cxx11")
add_links("SDL2", "SDL2_ttf")
add_cxflags("-g")
set_rundir("$(projectdir)")
