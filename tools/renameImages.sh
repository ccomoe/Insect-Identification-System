#!/bin/bash

# 提示用户输入目标目录
read -p "请输入要批量命名的文件夹路径 (支持相对路径和绝对路径): " input_path

# 如果用户未输入路径，则默认当前目录
if [ -z "$input_path" ]; then
    input_path="."
fi

# 将路径解析为绝对路径
target_path=$(realpath "$input_path" 2>/dev/null)

# 检查路径是否存在
if [ ! -d "$target_path" ]; then
    echo "路径不存在或不是文件夹，请检查输入！"
    exit 1
fi

# 定义命名函数
rename_files_in_folder() {
    local folder=$1
    local counter=0

    # 查找当前文件夹中的 png, jpg, jpeg 文件
    find "$folder" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | while read file; do
        # 获取文件扩展名并转换为小写
        extension=$(echo "${file##*.}" | tr 'A-Z' 'a-z')
        # 生成新文件名
        new_name="$folder/img$counter.$extension"
        # 检查是否有同名文件，避免覆盖
        while [ -e "$new_name" ]; do
            counter=$((counter + 1))
            new_name="$folder/img$counter.$extension"
        done
        # 重命名文件
        mv "$file" "$new_name"
        echo "已重命名: $file -> $new_name"
        # 增加计数器
        counter=$((counter + 1))
    done
}

# 遍历所有子目录并重命名
find "$target_path" -type d | while read dir; do
    echo "正在处理文件夹: $dir"
    rename_files_in_folder "$dir"
done

echo "批量命名完成！"
