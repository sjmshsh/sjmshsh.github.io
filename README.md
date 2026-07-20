# sjmshsh 的技术笔记

使用 Astro 构建、通过 GitHub Pages 发布的个人技术博客。

网站地址：<https://sjmshsh.github.io>

## 本地运行

```bash
nvm use
npm install
npm run dev
```

## 添加文章

在 `src/content/blog/` 中新增 Markdown 文件。文章图片建议放在同名目录中，并使用相对路径引用。

```markdown
---
title: '文章标题'
description: '文章摘要'
pubDate: '2026-07-21'
heroImage: './article-name/cover.jpg'
---

![图片说明](./article-name/diagram.png)
```

推送到 `main` 分支后，GitHub Actions 会自动构建并发布网站。
