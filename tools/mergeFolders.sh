#!/bin/bash

# 提示用户输入两个文件夹路径
read -p "请输入第一个文件夹路径 (支持相对路径和绝对路径): " folder1
read -p "请输入第二个文件夹路径 (支持相对路径和绝对路径): " folder2

# 如果路径为空，则退出
if [ -z "$folder1" ] || [ -z "$folder2" ]; then
    echo "两个路径都必须输入！"
    exit 1
fi

# 将路径解析为绝对路径
folder1=$(realpath "$folder1" 2>/dev/null)
folder2=$(realpath "$folder2" 2>/dev/null)

# 检查路径是否存在
if [ ! -d "$folder1" ]; then
    echo "路径 $folder1 不存在或不是文件夹！"
    exit 1
fi

if [ ! -d "$folder2" ]; then
    echo "路径 $folder2 不存在或不是文件夹！"
    exit 1
fi

# 创建合并函数
merge_folders() {
    local src=$1
    local dest=$2

    # 遍历源文件夹的所有内容
    for item in "$src"/*; do
        # 跳过不存在的路径 (防止空目录报错)
        [ ! -e "$item" ] && continue

        # 获取当前项的名称
        local name=$(basename "$item")

        # 如果是文件夹
        if [ -d "$item" ]; then
            # 如果目标文件夹中有同名文件夹，则递归合并
            if [ -d "$dest/$name" ]; then
                merge_folders "$item" "$dest/$name"
            else
                # 否则直接复制文件夹
                mkdir -p "$dest/$name"
                cp -r "$item/"* "$dest/$name"
            fi
        # 如果是文件
        elif [ -f "$item" ]; then
            # 如果目标文件夹中有同名文件，保留两个文件
            if [ -f "$dest/$name" ]; then
                # 为保留的文件生成新名称
                local new_name="$dest/${name%.*}_copy.${name##*.}"
                cp "$item" "$new_name"
            else
                # 否则直接复制文件
                cp "$item" "$dest/"
            fi
        fi
    done
}

# 开始合并
echo "正在合并文件夹 $folder1 和 $folder2 ..."
merge_folders "$folder1" "$folder2"
echo "文件夹合并完成！"
