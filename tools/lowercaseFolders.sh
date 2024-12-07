#!/bin/bash

# 提示用户输入路径（支持相对路径和绝对路径）
read -p "请输入要处理的路径 (默认为当前目录): " input_path

# 如果用户未输入路径，则使用当前目录
if [ -z "$input_path" ]; then
    input_path="."
fi

# 解析路径为绝对路径
target_path=$(realpath "$input_path" 2>/dev/null)

# 检查路径是否存在
if [ ! -d "$target_path" ]; then
    echo "路径不存在，请检查输入！"
    exit 1
fi

# 执行批量重命名
find "$target_path" -depth -type d | while read dir; do
    newdir=$(dirname "$dir")/$(basename "$dir" | tr 'A-Z' 'a-z')
    if [ "$dir" != "$newdir" ]; then
        mv "$dir" "$newdir"
    fi
done

echo "文件夹名称已处理完成！"
