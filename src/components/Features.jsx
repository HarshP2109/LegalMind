import React from 'react';

const featuresData = [
    {
        title: "Multilingual Conversational AI",
        description: "Support interactions in English and all other languages of the Indian Constitution.",
        img: "/images/feature-image-1.png", // Replace with actual image source
    },
    {
        title: "Optimized Legal Document",
        description: "Generating concise summaries of complex legal documents.",
        img: "/images/feature-image-2.png", // Replace with actual image source
    },
    {
        title: "Case Suggestions",
        description: "Provides suggestions from legal precedents and law books.",
        img: "/images/feature-image-3.png", // Replace with actual image source
    },
    {
        title: "Query Handling",
        description: "Responds to specific user questions based on the content of uploaded legal documents with accurate answers.",
        img: "/images/feature-image-4.png", // Replace with actual image source
    },
];

export default function Features() {
    return (
        <div className="px-8">
            <div 
            className="flex items-center flex-col mb-4">
                <div>
                    <h4
                    className='text-2xl font-semibold'>
                        Features of LegalMind
                    </h4>
                </div>
                <p className="lg:w-1/2 text-center">
                Explore features that boost your productivity. From document automation to advanced
                research, we've got the hard work covered.
                </p>
            </div>
            {/* Large Screens Layout */}
            <div className="hidden lg:grid lg:grid-cols-3 lg:grid-rows-2 lg:gap-4 z-slide-in">
                <div className="col-span-2 bg-white p-4 rounded-md">
                    <img src={featuresData[0].img} alt={featuresData[0].title} className="mt-2" />
                    <h3 className="text-xl font-bold">{featuresData[0].title}</h3>
                    <p>{featuresData[0].description}</p>
                </div>
                <div className="col-start-3 bg-white p-4 rounded-md">
                    <img src={featuresData[1].img} alt={featuresData[1].title} className="mt-2" />
                    <h3 className="text-xl font-bold">{featuresData[1].title}</h3>
                    <p>{featuresData[1].description}</p>
                </div>
                <div className="row-start-2 bg-white p-4 rounded-md">
                    <img src={featuresData[2].img} alt={featuresData[2].title} className="mt-2" />
                    <h3 className="text-xl font-bold">{featuresData[2].title}</h3>
                    <p>{featuresData[2].description}</p>
                </div>
                <div className="col-span-2 row-start-2 bg-white p-4 rounded-md">
                    <img src={featuresData[3].img} alt={featuresData[3].title} className="mt-2" />
                    <h3 className="text-xl font-bold">{featuresData[3].title}</h3>
                    <p>{featuresData[3].description}</p>
                </div>
            </div>

            {/* Medium Screens Layout (2 Columns) */}
            <div className="hidden md:grid md:grid-cols-2 md:gap-4 lg:hidden ">
                {featuresData.map((feature, index) => (
                    <div key={index} className="bg-white p-4 rounded-md">
                        <img src={feature.img} alt={feature.title} className="mt-2" />
                        <h3 className="text-xl font-bold">{feature.title}</h3>
                        <p>{feature.description}</p>
                    </div>
                ))}
            </div>

            {/* Small Screens Layout */}
            <div className="md:hidden grid grid-cols-1 gap-4 ">
                {featuresData.map((feature, index) => (
                    <div key={index} className="bg-white p-4 rounded-md">
                        <img src={feature.img} alt={feature.title} className="mt-2" />
                        <h3 className="text-xl font-bold">{feature.title}</h3>
                        <p>{feature.description}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
