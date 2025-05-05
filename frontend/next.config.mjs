/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { isServer }) => {
    // 서버 측에서 Webpack이 'canvas' 모듈을 번들링하려고 시도하지 않도록 설정
    if (isServer) {
      config.externals.push('canvas');
    }
    return config;
  },
};

export default nextConfig; 