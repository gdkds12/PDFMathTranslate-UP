/** @type {import('next').NextConfig} */
const nextConfig = {
  // reactStrictMode: true, // 필요에 따라 주석 해제 또는 설정 추가
  output: 'standalone', // <<< standalone 출력 모드 활성화

  webpack: (config, { isServer }) => {
    // 서버 측 빌드에서 canvas 모듈을 외부 모듈로 처리
    // react-pdf 사용 시 필요할 수 있음 (이전 오류 해결 과정에서 추가됨)
    if (isServer) {
      config.externals.push('canvas');
    }
    return config;
  },
};

export default nextConfig; 