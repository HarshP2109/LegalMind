import React, { useRef,useState} from 'react';
import Header from '../components/Header';
import HorizontalScroll from '../components/HorizontalScroll';
import Features from '../components/Features';
import ScrollToTopArrow from '../components/ScrollToTopArrow';
import AboutUs from '../components/AboutUs';
import Footer from '../components/Footer';
import ContactUs from '../components/ContactUs';
import ModalPoPUp from '../components/ModalPoPUp';

export default function Hero() {
    const section1Ref = useRef(null);
    const section2Ref = useRef(null);
    const section3Ref = useRef(null);
    const section4Ref = useRef(null);

    const scrollToSection = (sectionRef) => {
        sectionRef.current?.scrollIntoView({ behavior: "smooth" });
      };

      const [isModalOpen, setIsModalOpen] = useState(false);

      const openModal = () => {
          setIsModalOpen(true);
      };
  
      const closeModal = () => {
          setIsModalOpen(false);
      };
    return (
        <>
            <section className="mt-4"
            ref={section1Ref}
                >
                <Header scrollToSection={scrollToSection} section1Ref={section1Ref} section2Ref={section2Ref} section3Ref={section3Ref} section4Ref={section4Ref}
                 />
            </section>
            {/* Section for large screens */}
            <section className="hidden lg:block">
                <div className="grid grid-cols-1 sm:grid-cols-3 h-auto items-center overflow-hidden">
                    {/* Left Column */}
                    <div className="flex items-baseline justify-start text-4xl font-bold text-[#4C7766] ">
       
                    </div>
                    {/* Center Column with Image */}
                    <div className="flex justify-center items-end">
                        <img
                            src="/images/justice-lady.png"
                            alt="Justice Lady"
                            className="max-w-full h-auto  animate-right-slide-in"
                        />
                    </div>
                    {/* Right Column */}
                    <div className="flex items-center justify-center text-4xl font-bold text-[#4C7766]">
                    </div>
                </div>
                <div
                    className="z-20 top-[80px] left-[140px] absolute text-stroke text-[90px] font-bold text-transparent flex flex-col items-start -space-y-12 animate-left-slide-in">
                    <p>
                        FULLFILL 
                    </p>
                    <p>
                        YOUR
                    </p>
                </div>

                <div
                    className="z-20 top-[300px] right-[200px] absolute  text-[90px] font-bold flex flex-col items-start -space-y-12 animate-left-slide-in">
                    <p className="text-customGreen">
                        LEGAL 
                    </p>
                    <p 
                        className="text-stroke  text-transparent ml-10">
                        NEEDS
                    </p>
                </div>
                <div className="">

                <HorizontalScroll />
                </div>
            </section>
            {/* section for small screens */}
            <section 
                className="lg:hidden">
                <div
                    className="flex items-center justify-center -mt-8">
                <img 
                    src="/images/justice-lady.png"
                    alt="Justice Lady" 
                    className="ml-3 brightness animate-z-slide-in"/>
                </div>
                <div
                    className="z-20 top-[130px] left-4 absolute  lg:text-[90px] text-[70px] sm:text-[70px] font-bold flex flex-col items-start -space-y-12 animate-left-slide-in ">
                    <p className=" text-stroke  text-transparent ">
                        FULFILL 
                    </p>
                    <p 
                        className="text-customGreen">
                        YOUR
                    </p>
                </div>

                <div
                    className="z-20 top-[300px] right-4 absolute  lg:text-[90px] text-[70px] sm:text-[70px] font-bold flex flex-col items-start -space-y-12 animate-left-slide-in ">
                    <p className="text-customGreen  ">
                        LEGAL 
                    </p>
                    <p 
                        className=" text-stroke  text-transparent">
                        NEEDS
                    </p>
                </div>

                <div>
                    <HorizontalScroll/>
                </div>

            </section>
            <section
                className="mt-10"
                ref={section2Ref}>
                <Features/>
            </section>
            <section className="mt-10" ref={section3Ref}>
                <AboutUs/>
                <div className="bg-customGreen w-full py-4 flex items-center justify-center text-white">
                    READY?
                    <button className=" bg-white text-customGreen text-sm py-2 px-4 rounded-md ml-3"onClick={openModal}>
                        TRY NOW
                    </button>
                </div>
            </section>
            <section ref={section4Ref}>
                <ContactUs/>
            </section>
            <section>
                <Footer/>
            </section>
                <ScrollToTopArrow/>
                <ModalPoPUp isOpen={isModalOpen} closeModal={closeModal} />

        </>
    );
}
