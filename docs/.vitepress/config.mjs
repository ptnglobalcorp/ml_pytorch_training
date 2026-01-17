import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'ML with PyTorch',
  description: 'Hands-on PyTorch training for deep learning and neural networks',

  // Clean URLs (no .html extension)
  cleanUrls: true,

  // Ignore dead links for work-in-progress documentation
  ignoreDeadLinks: true,

  themeConfig: {
    // Site navigation
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Study Guide', link: '/README' },
      { text: 'Module 1', link: '/module-01/README' },
      { text: 'Module 2', link: '/module-02/README' },
      { text: 'Module 3', link: '/module-03/README' },
    ],

    // Sidebar configuration
    sidebar: {
      '/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Study Guide', link: '/README' },
          ]
        },
        {
          text: 'Module 1: Deep Learning Foundations',
          collapsed: false,
          items: [
            { text: 'Module Overview', link: '/module-01/README' },
            {
              text: 'Part 1: Conceptual Foundation',
              items: [
                { text: 'Introduction to DL', link: '/module-01/01-introduction-to-deep-learning' },
                { text: 'Neural Network Anatomy', link: '/module-01/02-neural-network-anatomy' },
                { text: 'Learning Methodology', link: '/module-01/03-learning-methodology' },
              ]
            },
            {
              text: 'Part 2: PyTorch Essentials',
              items: [
                { text: 'PyTorch Essentials', link: '/module-01/04-pytorch-essentials' },
                { text: 'Tensor Foundations', link: '/module-01/05-tensor-foundations' },
                { text: 'Tensor Operations', link: '/module-01/06-tensor-operations' },
                { text: 'Tensor Manipulation', link: '/module-01/07-tensor-manipulation' },
                { text: 'Exercises', link: '/module-01/exercises' },
              ]
            },
          ]
        },
        {
          text: 'Module 2: PyTorch Workflow Fundamentals',
          collapsed: true,
          items: [
            { text: 'Module Overview', link: '/module-02/README' },
            { text: 'Data Preparation', link: '/module-02/data-preparation' },
            { text: 'Building Models', link: '/module-02/building-models' },
            { text: 'Training Loop', link: '/module-02/training-loop' },
            { text: 'Model Persistence', link: '/module-02/model-persistence' },
          ]
        },
        {
          text: 'Module 3: Neural Network Classification',
          collapsed: true,
          items: [
            { text: 'Module Overview', link: '/module-03/README' },
            { text: 'Classification Basics', link: '/module-03/classification-basics' },
            { text: 'Architecture Design', link: '/module-03/architecture-design' },
            { text: 'Training & Evaluation', link: '/module-03/training-evaluation' },
            { text: 'Model Deployment', link: '/module-03/model-deployment' },
          ]
        },
      ]
    },

    // Social links
    socialLinks: [
      { icon: 'github', link: 'https://github.com/yourusername/ml-pytorch-training' }
    ],

    // Footer
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026-present'
    },

    // Edit link
    editLink: {
      pattern: 'https://github.com/yourusername/ml-pytorch-training/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    // Last updated text
    lastUpdated: {
      text: 'Last updated',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      }
    },

    // Search
    search: {
      provider: 'local'
    }
  },

  // Markdown configurations
  markdown: {
    // Line numbers in code blocks
    lineNumbers: true,

    // Language display
    config: (md) => {
      // Add custom markdown-it plugins if needed
      return md
    }
  },

  // Build optimizations
  vite: {
    build: {
      chunkSizeWarningLimit: 1000
    }
  }
})
